#pragma once
#include "common.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#include <map>
#include <string>
#include <string_view>

namespace conic {

inline void require(bool ok,const char* reason) { if(!ok)throw std::runtime_error(reason); }
inline uint16_t read16(const unsigned char* p) {return uint16_t(p[0])|(uint16_t(p[1])<<8);}
inline uint32_t read32(const unsigned char* p) {return uint32_t(read16(p))|(uint32_t(read16(p+2))<<16);}
inline uint64_t read64(const unsigned char* p) {return uint64_t(read32(p))|(uint64_t(read32(p+4))<<32);}

class Zip {
    struct Item {size_t compressed,len,offset;unsigned method;};
    unsigned char* data=nullptr;
    size_t size=0;
    std::map<std::string,Item> files;
public:
    explicit Zip(const char* name) {
        int fd=open(name,O_RDONLY|O_CLOEXEC);
        require(fd>=0,"cannot open NPZ");
        struct stat s{};
        if(fstat(fd,&s)!=0 || s.st_size<22 || s.st_size>(int64_t(1)<<30)) {
            close(fd);throw std::runtime_error("invalid NPZ size");
        }
        size=size_t(s.st_size);
        void* mem=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
        close(fd);
        require(mem!=MAP_FAILED,"cannot map NPZ");
        data=static_cast<unsigned char*>(mem);
        try {directory();} catch (...) {munmap(data,size);throw;}
    }
    ~Zip() {munmap(data,size);}
    Zip(const Zip&)=delete;
    Zip& operator=(const Zip&)=delete;

    size_t member_size(const std::string& name) const {
        const auto it=files.find(name+".npy");
        require(it!=files.end(),"missing NPZ member");
        return it->second.len;
    }

    void directory() {
        size_t k=size-22, lo=k>65535?k-65535:0;
        while(k>lo && !(read32(data+k)==0x06054b50 && k+22+read16(data+k+20)==size)) --k;
        require(read32(data+k)==0x06054b50,"missing ZIP directory");
        size_t n=read16(data+k+10), off=read32(data+k+16);
        if(n==0xffff || off==0xffffffff) {
            require(k>=20 && read32(data+k-20)==0x07064b50,"missing ZIP64 locator");
            size_t pos=read64(data+k-12);
            require(pos<=size-56 && read32(data+pos)==0x06064b50,"invalid ZIP64 directory");
            n=read64(data+pos+32);off=read64(data+pos+48);
        }
        for(size_t j=0;j<n;++j) {
            require(off<=size-46 && read32(data+off)==0x02014b50,"wrong ZIP directory entry");
            const auto *p=data+off;
            Item a{read32(p+20),read32(p+24),read32(p+42),read16(p+10)};
            size_t l=read16(p+28), extra=read16(p+30), end=off+46+l+extra;
            off=end+read16(p+32);
            require(off<=size,"invalid ZIP entry bounds");
            if(a.len==0xffffffff || a.compressed==0xffffffff || a.offset==0xffffffff) {
                const auto* e=p+46+l,*ep=data+end;
                while(ep-e>=4 && read16(e)!=1) {
                    const size_t el=read16(e+2);
                    require(size_t(ep-e)>=4+el,"invalid ZIP extra field");
                    e+=4+el;
                }
                require(ep-e>=4 && size_t(ep-e)>=4+read16(e+2),"missing ZIP64 field");
                const auto* lim=e+4+read16(e+2);
                e+=4;
                auto read=[&](size_t& out) {
                    if(out==0xffffffff) {require(lim-e>=8,"short ZIP64 field");out=read64(e);e+=8;}
                };
                read(a.len);read(a.compressed);read(a.offset);
            }
            files.emplace(std::string(reinterpret_cast<const char*>(p+46),l),a);
        }
    }

    std::vector<unsigned char> extract(const std::string& name) const {
        const auto it=files.find(name+".npy");
        require(it!=files.end(),"missing NPZ member");
        const auto& a=it->second;
        require(a.offset<=size-30 && a.len<=(size_t(1)<<30),"invalid NPZ member");
        const auto* p=data+a.offset;
        require(read32(p)==0x04034b50,"invalid ZIP local header");
        size_t start=a.offset+30+read16(p+26)+read16(p+28);
        require(start<=size && a.compressed<=size-start,"short NPZ member");
        std::vector<unsigned char> out(a.len);
        if(a.method==0) {
            require(a.compressed==a.len,"stored ZIP length mismatch");
            if(a.len) std::memcpy(out.data(),data+start,a.len);
        } else if(a.method==8) {
            z_stream z{};
            z.next_in=data+start;z.avail_in=uInt(a.compressed);
            z.next_out=out.data();z.avail_out=uInt(out.size());
            require(inflateInit2(&z,-MAX_WBITS)==Z_OK,"cannot start inflate");
            int ret=inflate(&z,Z_FINISH);
            inflateEnd(&z);
            require(ret==Z_STREAM_END && z.total_out==a.len,"invalid compressed NPY");
        } else throw std::runtime_error("unsupported ZIP compression");
        return out;
    }
};

inline std::string_view ltrim(std::string_view v) {
    size_t i=v.find_first_not_of(" \t\r\n");
    return i==v.npos?std::string_view{}:v.substr(i);
}

struct Npy {
    std::vector<unsigned char> buf;
    const unsigned char* data;
    size_t bytes;
    size_t count=1;
    char type,order;
    size_t width;
    bool scalar;
    explicit Npy(std::vector<unsigned char> a):buf(std::move(a)) {
        require(buf.size()>=10 && std::memcmp(buf.data(),"\x93NUMPY",6)==0,"not an NPY array");
        const int version=buf[6],prefix=version==1?10:12;
        require((version>=1 && version<=3) && buf.size()>=size_t(prefix),"unsupported NPY version");
        size_t len=version==1?read16(buf.data()+8):read32(buf.data()+8);
        require(len<=buf.size()-prefix,"short NPY header");
        std::string_view hdr(reinterpret_cast<const char*>(buf.data()+prefix),len);
        auto field=[&](const char* name) {
            size_t x=hdr.find(name), p=x!=hdr.npos?hdr.find(':',x):hdr.npos;
            require(p!=hdr.npos,"missing NPY header key");
            return ltrim(hdr.substr(p+1));
        };
        auto descr=field("descr");
        require(descr.size()>=5 && (descr[0]=='\''||descr[0]=='"'),"invalid NPY dtype");
        order=descr[1];type=descr[2];
        require(order=='<'||order=='>'||order=='|'||order=='=',"invalid NPY endianness");
        width=0;
        for(size_t b=3;b<descr.size() && descr[b]>='0' && descr[b]<='9';++b)
            width=10*width+descr[b]-'0';
        require(width>0 && width<=(size_t(1)<<28),"invalid NPY width");
        auto shape=field("shape");
        require(shape.size()>=2 && shape[0]=='(',"invalid NPY shape");
        shape=ltrim(shape.substr(1));
        scalar=!shape.empty() && shape[0]==')';
        if(!scalar) {
            count=0;
            require(!shape.empty() && shape[0]>='0' && shape[0]<='9',"expected NPY vector");
            while(!shape.empty() && shape[0]>='0' && shape[0]<='9') {
                count=10*count+size_t(shape[0]-'0');shape.remove_prefix(1);
                require(count<=(size_t(1)<<30),"NPY vector too long");
            }
            shape=ltrim(shape);
            require(!shape.empty() && shape[0]==',',"expected one-dimensional NPY");
            shape=ltrim(shape.substr(1));
            require(!shape.empty() && shape[0]==')',"expected one-dimensional NPY");
        }
        data=buf.data()+prefix+len;
        bytes=buf.size()-prefix-len;
        if(type=='U')width*=4;
        require(count<=bytes/width && count*width==bytes,"NPY payload length mismatch");
    }
    uint64_t integer(const unsigned char* p,size_t len) const {
        uint64_t k=0;
        if(order=='>') for(size_t i=0;i<len;++i)k=(k<<8)|p[i];
        else for(size_t i=0;i<len;++i)k|=uint64_t(p[i])<<(i*8);
        return k;
    }
    Vec floats() const {
        require(!scalar && type=='f' && width==8,"expected float64 vector");
        Vec out(count);
        if(order!='>') {if(bytes) std::memcpy(out.data(),data,bytes);}
        else for(size_t i=0;i<count;++i) {
            uint64_t u=integer(data+8*i,8);std::memcpy(out.data()+i,&u,8);
        }
        return out;
    }
    Int ints() const {
        require(!scalar && (type=='i'||type=='u') && width<=8,"expected integer vector");
        Int out(count);
        if(order!='>' && type=='i' && width==4) {
            if(bytes) std::memcpy(out.data(),data,bytes);
        } else for(size_t i=0;i<count;++i) {
            uint64_t k=integer(data+i*width,width);
            require(k<=uint64_t(std::numeric_limits<int>::max()),"unsupported index range");
            out[i]=int(k);
        }
        return out;
    }
    std::string text() const {
        require(scalar && type=='U',"expected scalar Unicode NPY");
        std::string out;
        for(size_t i=0;i<bytes;i+=4) {
            uint32_t x=integer(data+i,4);
            if(x==0)break;
            if(x<0x80)out.push_back(char(x));
            else if(x<0x800) {out.push_back(char(0xc0|(x>>6)));out.push_back(char(0x80|(x&63)));}
            else if(x<0x10000) {
                out.push_back(char(0xe0|(x>>12)));out.push_back(char(0x80|((x>>6)&63)));
                out.push_back(char(0x80|(x&63)));
            } else {
                out.push_back(char(0xf0|(x>>18)));out.push_back(char(0x80|((x>>12)&63)));
                out.push_back(char(0x80|((x>>6)&63)));out.push_back(char(0x80|(x&63)));
            }
        }
        return out;
    }
};

class NpzOutput {
    struct File { std::string name; uint32_t crc,len,start; };
    FILE* fp;
    std::vector<File> dir;
    uint32_t pos=0;
    void write(const void* p,size_t n) {
        if(n) require(std::fwrite(p,1,n,fp)==n,"cannot write output");
        pos+=n;
    }
    void w16(uint16_t v) { const unsigned char b[2]={uint8_t(v),uint8_t(v>>8)};write(b,2); }
    void w32(uint32_t v) { w16(v);w16(v>>16); }
    void member(const std::string& name,const std::string& type,const std::string& shape,
                const void* data,uint32_t bytes) {
        std::string header="{'descr': '"+type+"', 'fortran_order': False, 'shape': "+shape+", }";
        header.append((63-(header.size()+10)%64),' ');
        header+='\n';
        std::string npy("\x93NUMPY\x01\0",8);
        const size_t len=header.size();
        npy.push_back(char(len));npy.push_back(char(len>>8));npy+=header;
        uint32_t crc=crc32(0,reinterpret_cast<const unsigned char*>(npy.data()),npy.size());
        crc=crc32(crc,reinterpret_cast<const unsigned char*>(bytes?data:npy.data()),bytes);
        File f{name+".npy",crc,uint32_t(npy.size()+bytes),pos};
        w32(0x04034b50);w16(20);w16(0);w16(0);w16(0);w16(0x21);
        w32(f.crc);w32(f.len);w32(f.len);w16(f.name.size());w16(0);
        write(f.name.data(),f.name.size());write(npy.data(),npy.size());write(data,bytes);
        dir.push_back(std::move(f));
    }
public:
    explicit NpzOutput(const char* path):fp(std::fopen(path,"wb")) {
        require(fp!=nullptr,"cannot open result path");
    }
    ~NpzOutput(){if(fp)std::fclose(fp);}
    NpzOutput(const NpzOutput&)=delete;
    void vec(const char* name,const Vec& x) {
        member(name,"<f8","("+std::to_string(x.size())+",)",x.data(),8*x.size());
    }
    void metadata(const char* status) {
        std::string s=std::string("{\"schema_version\":1,\"status\":\"")+status+"\"}";
        std::vector<uint32_t> u(s.begin(),s.end());
        member("metadata_json","<U"+std::to_string(u.size()),"()",u.data(),4*u.size());
    }
    void finish() {
        uint32_t start=pos;
        for(const auto& f:dir) {
            w32(0x02014b50);w16(20);w16(20);w16(0);w16(0);w16(0);w16(0x21);
            w32(f.crc);w32(f.len);w32(f.len);
            w16(f.name.size());w16(0);w16(0);w16(0);w16(0);w32(0);w32(f.start);
            write(f.name.data(),f.name.size());
        }
        uint32_t len=pos-start;
        w32(0x06054b50);w16(0);w16(0);w16(dir.size());w16(dir.size());w32(len);w32(start);w16(0);
        bool ok=std::fclose(fp)==0;fp=nullptr;require(ok,"cannot finish output");
    }
};

struct OwnedCSC {
    Int p,i;
    Vec a;
    void read(const Zip& zip,const std::string& name,int rows,int cols) {
        p=Npy(zip.extract(name+"_indptr")).ints();
        i=Npy(zip.extract(name+"_indices")).ints();
        a=Npy(zip.extract(name+"_data")).floats();
        require(p.size()==size_t(cols)+1 && p.front()==0 && size_t(p.back())==i.size() &&
            i.size()==a.size(),"invalid CSC arrays");
        int pos=0;
        std::vector<std::pair<int,double>> tmp;
        for(int j=0;j<cols;++j) {
            const int lo=p[j], hi=p[j+1];
            require(lo>=0&&hi>=lo&&size_t(hi)<=i.size(),"invalid CSC pointer");
            bool sorted=true;
            for(int k=lo;k<hi;++k) {
                require(i[k]>=0 && i[k]<rows,"invalid CSC row");
                if(k>lo && i[k]<=i[k-1])sorted=false;
            }
            p[j]=pos;
            if(sorted) {
                for(int k=lo;k<hi;++k) if(a[k]!=0) {i[pos]=i[k];a[pos++]=a[k];}
            } else {
                tmp.clear();for(int k=lo;k<hi;++k)tmp.emplace_back(i[k],a[k]);
                std::stable_sort(tmp.begin(),tmp.end(),[](auto x,auto y){return x.first<y.first;});
                for(size_t k=0;k<tmp.size();) {
                    int r=tmp[k].first;
                    double v=tmp[k++].second;
                    while(k<tmp.size() && tmp[k].first==r) v+=tmp[k++].second;
                    if(v!=0){i[pos]=r;a[pos++]=v;}
                }
            }
        }
        p[cols]=pos;i.resize(pos);a.resize(pos);
    }
};

} // namespace conic
