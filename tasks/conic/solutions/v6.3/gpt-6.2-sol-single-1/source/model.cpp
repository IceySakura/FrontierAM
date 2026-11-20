#include "model.hpp"

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
// Stable zlib C ABI; the image ships the runtime library, not zlib-dev.
struct ZipStream {
    const unsigned char* next_in;
    unsigned int avail_in;
    unsigned long total_in;
    unsigned char* next_out;
    unsigned int avail_out;
    unsigned long total_out;
    char* msg;
    void* state;
    void* (*zalloc)(void*,unsigned int,unsigned int);
    void (*zfree)(void*,void*);
    void* opaque;
    int data_type;
    unsigned long adler,reserved;
};
const char* zlibVersion();
int inflateInit2_(ZipStream*,int,const char*,int);
int inflate(ZipStream*,int);
int inflateEnd(ZipStream*);
unsigned long crc32(unsigned long,const unsigned char*,unsigned int);
}

namespace {
using Bytes = std::vector<unsigned char>;
using Byte = unsigned char;
using View = std::string_view;
void check(bool v,const char* why) { if(!v) throw std::runtime_error(why); }
template<class T> T load(const Byte* p) {
    T v;
    std::memcpy(&v,p,sizeof(v));
    return v;
}
uint16_t u16(const Byte* p) {return load<uint16_t>(p);}
uint32_t u32(const Byte* p) {return load<uint32_t>(p);}
template<class T> void put(Bytes& p,T v) {
    auto n=p.size(); p.resize(n+sizeof(T)); std::memcpy(p.data()+n,&v,sizeof(T));
}

class ZipReader {
    const Byte* file = nullptr;
    size_t len=0;
    struct Member {size_t start, packed, raw; int method;};
    std::map<std::string,Member,std::less<>> members;
public:
    explicit ZipReader(const char* path) {
        int fd=open(path,O_RDONLY|O_CLOEXEC);
        check(fd>=0,"cannot open input");
        struct stat st{};
        if(fstat(fd,&st)<0 || st.st_size<22){close(fd); throw std::runtime_error("bad input size");}
        len=st.st_size;
        file=(const Byte*)mmap(nullptr,len,PROT_READ,MAP_PRIVATE,fd,0);
        close(fd);
        check(file!=MAP_FAILED,"cannot map input");
        try {
            size_t e=len-22, mn=e>65535?e-65535:0;
            while(e>mn && !(u32(file+e)==0x06054b50 && e+22+u16(file+e+20)==len)) --e;
            check(u32(file+e)==0x06054b50,"missing ZIP central directory");
            size_t count=u16(file+e+10), pos=u32(file+e+16);
            if((count==65535 || pos==0xffffffff) && e>=20 && u32(file+e-20)==0x07064b50) {
                size_t z=load<uint64_t>(file+e-12);
                check(z+56<=e && u32(file+z)==0x06064b50,"ZIP64 central directory");
                count=load<uint64_t>(file+z+32);
                pos=load<uint64_t>(file+z+48);
            }
            for(size_t i=0;i<count;++i) {
                check(pos+46<=e,"truncated ZIP directory");
                auto p=file+pos;
                check(u32(p)==0x02014b50,"invalid ZIP directory");
                size_t n=u16(p+28), extra=u16(p+30), comment=u16(p+32);
                size_t packed=u32(p+20), raw=u32(p+24), off=u32(p+42);
                pos+=46+n+extra+comment;
                check(pos<=len,"truncated ZIP entry");
                if(packed==0xffffffff || raw==0xffffffff || off==0xffffffff) {
                    auto z=p+46+n, end=z+extra;
                    while(z+4<=end && u16(z)!=1) z+=4+u16(z+2);
                    check(z+4<=end,"missing ZIP64 size");
                    z+=4;
                    auto next=[&](){check(z+8<=end,"short ZIP64 size"); size_t t=load<uint64_t>(z); z+=8; return t;};
                    if(raw==0xffffffff) raw=next();
                    if(packed==0xffffffff) packed=next();
                    if(off==0xffffffff) off=next();
                }
                check(off+30<=len && u32(file+off)==0x04034b50,"invalid ZIP local header");
                const auto q=file+off;
                size_t start=off+30+u16(q+26)+u16(q+28);
                check(start<=len && packed<=len-start,"truncated ZIP data");
                members.emplace(std::string((const char*)p+46,n),Member{start,packed,raw,u16(p+10)});
            }
        } catch(...) { munmap((void*)file,len); throw; }
    }
    ~ZipReader() {munmap((void*)file,len);}
    struct Npy {
        Bytes buf;
        const Byte* data=nullptr;
        size_t size=0, stride=0;
        char kind=0;
        bool swap=false;
        template<typename T> T loadval(size_t i) const {
            T v=load<T>(data+i*stride);
            if(swap) {
                auto b=reinterpret_cast<Byte*>(&v);
                std::reverse(b,b+sizeof(T));
            }
            return v;
        }
        Vec floats() const {
            check(kind=='f' && stride==8,"expected float64");
            Vec v(size);
            if(!swap && size) std::memcpy(v.data(),data,size*8);
            else for(size_t i=0;i<size;++i) v[i]=loadval<double>(i);
            return v;
        }
        Ind ints() const {
            check(kind=='i' || kind=='u',"expected integer array");
            Ind v(size);
            if(!swap && stride==4 && size && kind=='i') std::memcpy(v.data(),data,size*4);
            else for(size_t i=0;i<size;++i){
                int64_t n= stride==8 ? loadval<int64_t>(i) : stride==4 ? loadval<uint32_t>(i) :
                           stride==2 ? loadval<uint16_t>(i) : stride==1 ? loadval<uint8_t>(i) : -1;
                check(n>=0 && n<=INT32_MAX,"index out of range");
                v[i]=(int)n;
            }
            return v;
        }
        std::string text() const {
            check((kind=='U' || kind=='S') && size==1,"expected metadata string");
            if(kind=='S') return std::string((const char*)data,strnlen((const char*)data,stride));
            std::string out;
            // Preserve real Unicode in string values, leaving JSON syntax untouched.
            out.reserve(stride/4);
            for(size_t i=0;i<stride;i+=4){
                uint32_t c=load<uint32_t>(data+i);
                if(swap) c=__builtin_bswap32(c);
                if(c==0) break;
                if(c<0x80) out+=char(c);
                else if(c<0x800) {out+=char(0xc0|(c>>6)); out+=char(0x80|(c&63));}
                else if(c<0x10000) {out+=char(0xe0|(c>>12)); out+=char(0x80|((c>>6)&63)); out+=char(0x80|(c&63));}
                else {out+=char(0xf0|(c>>18)); out+=char(0x80|((c>>12)&63)); out+=char(0x80|((c>>6)&63)); out+=char(0x80|(c&63));}
            }
            return out;
        }
    };
    Npy npy(const std::string& name) const {
        auto it=members.find(name+".npy");
        check(it!=members.end(),"missing NPY member");
        auto [start,packed,raw,method]=it->second;
        check(raw<=1073741824,"NPY member exceeds resource limit");
        Npy n;
        const Byte* p=file+start;
        if(method!=0) {
            check(method==8,"unsupported ZIP compression");
            n.buf.resize(raw);
            ZipStream stream{};
            stream.next_in=p;stream.avail_in=packed;
            stream.next_out=n.buf.data();stream.avail_out=raw;
            check(inflateInit2_(&stream,-15,zlibVersion(),sizeof(stream))==0,"zlib initialization");
            int result=inflate(&stream,4);
            inflateEnd(&stream);
            check(result==1 && stream.total_out==raw,"invalid deflate member");
            p=n.buf.data();
        } else check(packed==raw,"invalid stored member");
        check(raw>=10 && std::memcmp(p,"\x93NUMPY",6)==0,"invalid NPY member");
        const int start_header=p[6]==1 ? 10 : 12;
        check(raw>=size_t(start_header),"missing NPY header");
        size_t hlen=start_header==10?u16(p+8):u32(p+8);
        check(start_header+hlen<=raw,"truncated NPY header");
        View h((const char*)p+start_header,hlen);
        size_t ds=h.find("descr");
        check(ds!=View::npos,"missing dtype");
        ds=h.find(':',ds)+1;
        ds=h.find_first_of("'\"",ds)+1;
        check(ds>0 && ds+2<h.size(),"invalid dtype");
        n.swap=h[ds]=='>';
        n.kind=h[ds+1];
        auto [end,err]=std::from_chars(h.data()+ds+2,h.data()+h.size(),n.stride);
        check(err==std::errc() && n.stride,"invalid NPY element size");
        if(n.kind=='U') n.stride*=4;
        n.size=(raw-start_header-hlen)/n.stride;
        check(n.size*n.stride==raw-start_header-hlen,"invalid NPY data size");
        n.data=p+start_header+hlen;
        return n;
    }
};

// Small JSON cursor: metadata values other than the cone list and offset are skipped.
class Json {
    View s;
    size_t p=0;
public:
    explicit Json(View data) : s(data) {}
    void ws() {p=s.find_first_not_of(" \t\r\n",p); if(p==View::npos) p=s.size();}
    char peek(){ws(); return p<s.size()?s[p]:'\0';}
    bool take(char c){if(peek()!=c) return false; ++p; return true;}
    void expect(char c){check(take(c),"invalid metadata JSON");}
    std::string string() {
        expect('"'); std::string t;
        while(p<s.size()) {
            char c=s[p++];
            if(c=='"') return t;
            if(c=='\\' && p<s.size()) {
                c=s[p++];
                if(c=='u' && p+4<=s.size()){
                    unsigned cp=0;
                    auto [end,ec]=std::from_chars(s.data()+p,s.data()+p+4,cp,16);
                    check(end==s.data()+p+4 && ec==std::errc(),"invalid JSON escape");
                    p+=4;
                    if(cp<128)c=char(cp);
                    else {
                        if(cp<2048)t+=char(0xc0|(cp>>6));
                        else {t+=char(0xe0|(cp>>12));t+=char(0x80|((cp>>6)&63));}
                        c=char(0x80|(cp&63));
                    }
                }
            }
            t+=c;
        }
        throw std::runtime_error("unterminated metadata string");
    }
    double number() {
        ws(); double x=0;
        auto [end,err]=std::from_chars(s.data()+p,s.data()+s.size(),x);
        check(err==std::errc(),"invalid metadata number"); p=end-s.data(); return x;
    }
    void skip(int depth=0) {
        check(depth<200,"metadata JSON nesting");
        char v=peek();
        if(v=='"'){string();return;}
        if(take('{')){
            if(take('}')) return;
            do {string();expect(':');skip(depth+1);} while(take(','));
            expect('}');return;
        }
        if(take('[')){
            if(take(']')) return;
            do {skip(depth+1);} while(take(','));
            expect(']');return;
        }
        auto j=p; while(p<s.size() && std::strchr(",]} \t\r\n",s[p])==nullptr) ++p;
        check(j!=p,"invalid metadata value");
    }
};

void metadata(const std::string& text,Model& m) {
    Json in(text);
    in.expect('{');
    do {
        std::string key=in.string();
        in.expect(':');
        if(key=="schema_version") check(in.number()==1,"unsupported schema");
        else if(key=="objective_offset") m.offset=in.number();
        else if(key=="cones") {
            in.expect('[');
            if(in.take(']')) continue;
            do {
                in.expect('{');
                ConeSpec b{-1,0};
                do {
                    auto k=in.string(); in.expect(':');
                    if(k=="type"){
                        auto t=in.string();
                        if(t=="zero") b.kind=Zero;
                        else if(t=="nonnegative") b.kind=Orthant;
                        else if(t=="soc") b.kind=Lorentz;
                        else if(t=="psd") b.kind=Semidefinite;
                        else if(t=="exponential") b.kind=Exponential;
                    } else if(k=="dim") b.dim=(int)in.number();
                    else in.skip();
                } while(in.take(','));
                in.expect('}');
                check(b.kind>=0 && b.dim>0,"invalid cone");
                m.cones.push_back(b);
            } while(in.take(','));
            in.expect(']');
        } else in.skip();
    } while(in.take(','));
    in.expect('}');
}

class ZipWriter {
    FILE* file;
    Bytes central;
    uint32_t pos=0;
    uint16_t num=0;
    void write(const void* d,size_t n) {
        if(n && fwrite(d,1,n,file)!=n) throw std::runtime_error("cannot write result");
        pos+=n;
    }
public:
    explicit ZipWriter(const char* path):file(fopen(path,"wb")) {
        check(file!=nullptr,"cannot open result");
    }
    ~ZipWriter(){fclose(file);}
    void npy(const std::string& name,const std::string& descr,const std::string& shape,
             const void* data,size_t size){
        std::string h="{'descr': '"+descr+"', 'fortran_order': False, 'shape': "+shape+", }";
        h.append((64-((10+h.size()+1)%64))%64,' ');
        h+='\n';
        Bytes buf={0x93,'N','U','M','P','Y',1,0};
        put<uint16_t>(buf,h.size());
        buf.insert(buf.end(),h.begin(),h.end());
        std::string member=name+".npy";
        auto ts=uint32_t(size+buf.size());
        auto crc=uint32_t(crc32(crc32(0,buf.data(),buf.size()),(const Byte*)data,size));
        // crc32(NULL,0) is a reset operation in zlib, not an empty update.
        if(!size) crc=uint32_t(crc32(0,buf.data(),buf.size()));
        uint32_t off=pos;
        Bytes e;
        put<uint32_t>(e,0x04034b50);
        put<uint16_t>(e,20);
        for(int i=0;i<4;++i) put<uint16_t>(e,0);
        put<uint32_t>(e,crc); put<uint32_t>(e,ts); put<uint32_t>(e,ts);
        put<uint16_t>(e,member.size()); put<uint16_t>(e,0);
        write(e.data(),e.size());write(member.data(),member.size());
        write(buf.data(),buf.size());write(data,size);
        put<uint32_t>(central,0x02014b50);
        put<uint16_t>(central,20);
        central.insert(central.end(),e.begin()+4,e.end());
        for(int i=0;i<3;++i) put<uint16_t>(central,0);
        put<uint32_t>(central,0); put<uint32_t>(central,off);
        central.insert(central.end(),member.begin(),member.end());
        ++num;
    }
    void vec(const char* name,const Vec& x){
        npy(name,"<f8","("+std::to_string(x.size())+",)",x.data(),x.size()*8);
    }
    void finish() {
        uint32_t off=pos;
        write(central.data(),central.size());
        Bytes e;
        put<uint32_t>(e,0x06054b50);
        put<uint16_t>(e,0); put<uint16_t>(e,0);
        put<uint16_t>(e,num); put<uint16_t>(e,num);
        put<uint32_t>(e,central.size()); put<uint32_t>(e,off);
        put<uint16_t>(e,0);
        write(e.data(),e.size());
        check(fflush(file)==0,"cannot flush result");
    }
};
}

void Sparse::compact() {
    // Fast path for canonical CSC; sort and accumulate only irregular columns.
    if(x.empty())return;
    bool clean=true;
    for(int j=0;j<cols && clean;++j)
        for(int k=p[j],last=-1;k<p[j+1];++k)
            if(i[k]<=last || x[k]==0){clean=false;break;} else last=i[k];
    if(clean) return;
    std::vector<std::pair<int,double>> buffer;
    int dst=0;
    for(int j=0;j<cols;++j){
        int lo=p[j],hi=p[j+1];p[j]=dst;
        bool sorted=true;
        for(int k=lo+1;k<hi;++k) if(i[k]<i[k-1]) {sorted=false;break;}
        if(!sorted){
            buffer.clear();
            for(int k=lo;k<hi;++k) buffer.emplace_back(i[k],x[k]);
            std::stable_sort(buffer.begin(),buffer.end(),[](auto&a,auto&b){return a.first<b.first;});
            for(int k=lo;k<hi;++k){i[k]=buffer[k-lo].first;x[k]=buffer[k-lo].second;}
        }
        for(int k=lo;k<hi;){
            int r=i[k];
            double s=x[k++];
            while(k<hi && i[k]==r) s+=x[k++];
            if(s!=0){i[dst]=r;x[dst++]=s;}
        }
    }
    p[cols]=dst; i.resize(dst); x.resize(dst);
}
Sparse Sparse::transpose() const {
    Sparse a(cols,rows); a.i.resize(i.size());a.x.resize(x.size());
    for(int v:i) ++a.p[v+1];
    for(int j=0;j<rows;++j) a.p[j+1]+=a.p[j];
    auto work=a.p;
    for(int j=0;j<cols;++j) for(int k=p[j];k<p[j+1];++k){
        int z=work[i[k]]++;a.i[z]=j;a.x[z]=x[k];
    }
    return a;
}
void Sparse::mul(const double* a,double* y) const {
    if(rows) std::fill(y,y+rows,0.0);
    for(int j=0;j<cols;++j){
        double s=a[j];
        for(int k=p[j];k<p[j+1];++k) y[i[k]]+=s*x[k];
    }
}
void Sparse::tmul(const double* a,double* y) const {
    for(int j=0;j<cols;++j) {
        double s=0;
        for(int k=p[j];k<p[j+1];++k) s+=x[k]*a[i[k]];
        y[j]=s;
    }
}
void Sparse::scale(const Vec& c,const Vec* r) {
    if(r) for(int j=0;j<cols;++j) for(int k=p[j];k<p[j+1];++k) x[k]*=c[j]*(*r)[i[k]];
    else for(int j=0;j<cols;++j) for(int k=p[j];k<p[j+1];++k) x[k]*=c[j];
}
void Sparse::norms(Vec& c,Vec* r) const {
    if(r) for(int j=0;j<cols;++j) for(int k=p[j];k<p[j+1];++k) {
        double v=std::abs(x[k]); c[j]=std::max(c[j],v); (*r)[i[k]]=std::max((*r)[i[k]],v);
    } else for(int j=0;j<cols;++j) for(int k=p[j];k<p[j+1];++k) c[j]=std::max(c[j],std::abs(x[k]));
}
Sparse symmetric_part(const Sparse& p) {
    if(!p.nnz()) return p;
    auto t=p.transpose();
    if(p.p==t.p && p.i==t.i && p.x==t.x) return p;
    Sparse a(p.rows,p.cols);
    a.i.reserve(std::max(p.nnz(),t.nnz())); a.x.reserve(a.i.capacity());
    for(int j=0;j<p.cols;++j){
        int k=p.p[j], l=t.p[j], e=p.p[j+1], f=t.p[j+1];
        while(k<e || l<f) {
            int u=k<e?p.i[k]:INT32_MAX, v=l<f?t.i[l]:INT32_MAX;
            int idx=std::min(u,v); double val=0;
            if(u==idx) val+=0.5*p.x[k++];
            if(v==idx) val+=0.5*t.x[l++];
            if(val!=0){a.i.push_back(idx);a.x.push_back(val);}
        }
        a.p[j+1]=a.nnz();
    }
    return a;
}
Model read_npz(const char* path) {
    ZipReader in(path);
    Model m;
    metadata(in.npy("metadata_json").text(),m);
    m.q=in.npy("q").floats();
    m.b=in.npy("b").floats();
    auto read=[&](const char* prefix,int rows,int cols){
        const std::string s=prefix;
        Sparse a; a.rows=rows; a.cols=cols;
        a.p=in.npy(s+"_indptr").ints();
        a.i=in.npy(s+"_indices").ints();
        a.x=in.npy(s+"_data").floats();
        check(a.p.size()==(size_t)cols+1 && a.i.size()==a.x.size() && a.p.back()==a.nnz(),
              "inconsistent CSC dimensions");
        a.compact();
        return a;
    };
    m.P=read("P",m.n(),m.n()); m.A=read("A",m.m(),m.n());
    int64_t total=0;
    for(auto c:m.cones) total+=c.width();
    check(total==m.m(),"inconsistent cone dimensions");
    return m;
}
void write_npz(const char* path,const Result& r) {
    ZipWriter out(path);
    const char* names[]={"unresolved","solved","primal_infeasible","dual_infeasible"};
    const auto s=std::string("{\"schema_version\":1,\"status\":\"")+names[r.status]+"\"}";
    std::vector<uint32_t> t(s.begin(),s.end());
    out.npy("metadata_json","<U"+std::to_string(t.size()),"()",t.data(),t.size()*4);
    if(r.status==Solved || r.status==DualInfeasible) {out.vec("x",r.x);out.vec("s",r.s);}
    if(r.status==Solved || r.status==PrimalInfeasible) out.vec("z",r.z);
    out.finish();
}
