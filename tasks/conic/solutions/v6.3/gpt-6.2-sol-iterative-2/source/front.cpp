#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <unistd.h>

// Stable C ABIs of the system libarchive and Jansson runtimes. The build
// deliberately does not require development packages for either library.
extern "C" {
struct archive;
struct archive_entry;
archive* archive_read_new();
int archive_read_support_format_zip(archive*);
int archive_read_open_filename(archive*,const char*,std::size_t);
int archive_read_next_header(archive*,archive_entry**);
long archive_read_data(archive*,void*,std::size_t);
int archive_read_data_skip(archive*);
int archive_read_free(archive*);
archive* archive_write_new();
int archive_write_set_format_zip(archive*);
int archive_write_set_format_option(archive*,const char*,const char*,const char*);
int archive_write_set_bytes_per_block(archive*,int);
int archive_write_open_filename(archive*,const char*);
int archive_write_header(archive*,archive_entry*);
long archive_write_data(archive*,const void*,std::size_t);
int archive_write_close(archive*);
int archive_write_free(archive*);
archive_entry* archive_entry_new();
const char* archive_entry_pathname(archive_entry*);
int64_t archive_entry_size(archive_entry*);
void archive_entry_set_pathname(archive_entry*,const char*);
void archive_entry_set_filetype(archive_entry*,unsigned);
void archive_entry_set_perm(archive_entry*,int);
void archive_entry_set_size(archive_entry*,int64_t);
void archive_entry_free(archive_entry*);
struct json_t;
json_t* json_loadb(const char*,std::size_t,std::size_t,void*);
json_t* json_object_get(const json_t*,const char*);
std::size_t json_array_size(const json_t*);
json_t* json_array_get(const json_t*,std::size_t);
const char* json_string_value(const json_t*);
int64_t json_integer_value(const json_t*);
double json_number_value(const json_t*);
void json_delete(json_t*);
}

namespace {

using Clock=std::chrono::steady_clock;
using Time=Clock::time_point;
using V=std::vector<double>;
using I=std::vector<int>;
constexpr double inf=std::numeric_limits<double>::infinity();
constexpr double eps=9.3e-9;

void require(bool c) {
    if (!c) throw std::runtime_error("input not supported by native front end");
}

struct Npy {
    std::vector<char> bytes;
    std::string type;
    std::size_t begin=0;

    void parse() {
        require(bytes.size()>=10 && std::memcmp(bytes.data(),"\x93NUMPY",6)==0);
        const auto* p=reinterpret_cast<const unsigned char*>(bytes.data());
        std::size_t len=p[8]+256u*p[9], base=10;
        if (p[6]>1) {
            require(bytes.size()>=12);
            len+=65536u*p[10]+16777216u*p[11];
            base=12;
        }
        require(base+len<=bytes.size());
        begin=base+len;
        std::string_view head(bytes.data()+base,len);
        auto k=head.find("descr");
        require(k!=head.npos);
        auto a=head.find_first_of("\"'",head.find(':',k)+1);
        require(a!=head.npos);
        auto b=head.find(head[a],a+1);
        require(b!=head.npos);
        type=std::string(head.substr(a+1,b-a-1));
    }

    V floats() const {
        require(type=="<f8" || type=="=f8");
        std::size_t size=bytes.size()-begin;
        require(size%8==0);
        V v(size/8);
        if (size) std::memcpy(v.data(),bytes.data()+begin,size);
        return v;
    }

    I ints() const {
        std::size_t size=bytes.size()-begin;
        if (type=="<i4" || type=="=i4") {
            require(size%4==0);
            I v(size/4);
            if (size) std::memcpy(v.data(),bytes.data()+begin,size);
            return v;
        }
        require((type=="<i8" || type=="=i8") && size%8==0);
        I v(size/8);
        for (std::size_t i=0;i<v.size();++i) {
            int64_t w;
            std::memcpy(&w,bytes.data()+begin+8*i,8);
            require(w>=0 && w<=std::numeric_limits<int>::max());
            v[i]=static_cast<int>(w);
        }
        return v;
    }

    std::string utf8() const {
        require(type.size()>2 && type[0]=='<' && type[1]=='U');
        std::string out;
        out.reserve((bytes.size()-begin)/4);
        for (std::size_t i=begin;i+4<=bytes.size();i+=4) {
            uint32_t w;
            std::memcpy(&w,bytes.data()+i,4);
            if (!w) break;
            if (w<0x80) out.push_back(char(w));
            else if (w<0x800) {
                out.push_back(char(0xc0|(w>>6)));
                out.push_back(char(0x80|(w&63)));
            } else {
                if (w>=0x10000) out.push_back(char(0xf0|(w>>18)));
                out.push_back(char((w>=0x10000?0x80:0xe0)|((w>>12)&(w>=0x10000?63:15))));
                out.push_back(char(0x80|((w>>6)&63)));
                out.push_back(char(0x80|(w&63)));
            }
        }
        return out;
    }
};

std::unordered_map<std::string,Npy> read_npz(const char* path) {
    std::unique_ptr<archive,decltype(&archive_read_free)> ar(archive_read_new(),&archive_read_free);
    require(ar && archive_read_support_format_zip(ar.get())==0);
    require(archive_read_open_filename(ar.get(),path,65536)==0);
    std::unordered_map<std::string,Npy> out;
    std::size_t loaded=0;
    archive_entry* e=nullptr;
    for (int ret;(ret=archive_read_next_header(ar.get(),&e))!=1;) {
        require(ret==0);
        std::string name=archive_entry_pathname(e);
        if (name!="q.npy" && name!="b.npy" && name!="metadata_json.npy"
            && name!="P_data.npy" && name!="A_data.npy"
            && name!="P_indices.npy" && name!="A_indices.npy"
            && name!="P_indptr.npy" && name!="A_indptr.npy") {
            archive_read_data_skip(ar.get());
            continue;
        }
        auto size=archive_entry_size(e);
        require(size>=0 && size<300000000);
        if (name=="A_data.npy") {
            auto b=out.find("b.npy");
            if (b!=out.end()) {
                auto bsize=b->second.bytes.size()-b->second.begin;
                // A singleton-per-row matrix cannot have more entries than
                // b, apart from explicit zeros or duplicates handled later.
                require(static_cast<std::size_t>(size)<=bsize+512);
            }
        }
        if (name=="P_data.npy") {
            auto q=out.find("q.npy");
            if (q!=out.end()) {
                auto qsize=q->second.bytes.size()-q->second.begin;
                require(static_cast<std::size_t>(size)<=16*qsize+512);
            }
        }
        loaded+=size;
        require(loaded<550000000);
        auto& arr=out[name];
        arr.bytes.resize(size);
        std::size_t pos=0;
        while (pos<arr.bytes.size()) {
            auto r=archive_read_data(ar.get(),arr.bytes.data()+pos,arr.bytes.size()-pos);
            require(r>0);
            pos+=r;
        }
        arr.parse();
        if (name=="b.npy") {
            auto a=out.find("A_data.npy");
            if (a!=out.end())
                require(a->second.bytes.size()-a->second.begin<=arr.bytes.size()-arr.begin+512);
        }
    }
    return out;
}

struct CSC {
    I p,i;
    V v;
    int cols=0, rows=0;

    void clean() {
        require(p.size()==std::size_t(cols)+1 && p[0]==0 && p.back()>=0
                && std::size_t(p.back())==v.size() && v.size()==i.size());
        bool simple=true;
        for (int j=0;j<cols;++j) {
            require(p[j]<=p[j+1] && p[j+1]<=p.back());
            int last=-1;
            for (int t=p[j];t<p[j+1];++t) {
                require(i[t]>=0 && i[t]<rows);
                simple &= i[t]>last && v[t]!=0.;
                last=i[t];
            }
        }
        if (simple) return;
        std::vector<std::pair<int,double>> block;
        int pos=0, start=0;
        for (int j=0;j<cols;++j) {
            int end=p[j+1];
            p[j]=pos;
            block.clear();
            for (int k=start;k<end;++k)
                block.emplace_back(i[k],v[k]);
            std::stable_sort(block.begin(),block.end(),
                             [](auto a,auto b){return a.first<b.first;});
            for (int k=0; k<static_cast<int>(block.size());) {
                int r=block[k].first;
                double sum=block[k++].second;
                for (; k<static_cast<int>(block.size()) && block[k].first==r; ++k)
                    sum+=block[k].second;
                if (sum!=0.) { i[pos]=r; v[pos++]=sum; }
            }
            start=end;
        }
        p[cols]=pos;
        i.resize(pos); v.resize(pos);
    }

    V mv(const V& x) const {
        V y(rows);
        for (int j=0;j<cols;++j) {
            double vj=x[j];
            for (int t=p[j];t<p[j+1];++t) y[i[t]]+=v[t]*vj;
        }
        return y;
    }

    V tmv(const V& x) const {
        V y(cols);
        for (int j=0;j<cols;++j)
            for (int t=p[j];t<p[j+1];++t) y[j]+=v[t]*x[i[t]];
        return y;
    }
};

enum Type { Zero, Pos, Soc, Exp, Psd };
struct Cone { Type kind; int dim, start, end; };

struct Problem {
    CSC A,P;
    V q,b;
    int n,m;
    double offset=0.;
    std::vector<Cone> K;
    bool linear=true;

    explicit Problem(std::unordered_map<std::string,Npy>& f) {
        q=f.at("q.npy").floats();
        b=f.at("b.npy").floats();
        n=q.size(); m=b.size();
        A={f.at("A_indptr.npy").ints(), f.at("A_indices.npy").ints(),
           f.at("A_data.npy").floats(),n,m};
        P={f.at("P_indptr.npy").ints(), f.at("P_indices.npy").ints(),
           f.at("P_data.npy").floats(),n,n};
        A.clean(); P.clean();
        auto meta=f.at("metadata_json.npy").utf8();
        std::unique_ptr<json_t,decltype(&json_delete)> root(json_loadb(meta.data(),meta.size(),0,nullptr),&json_delete);
        require(bool(root));
        offset=json_number_value(json_object_get(root.get(),"objective_offset"));
        const json_t* cones=json_object_get(root.get(),"cones");
        require(cones!=nullptr);
        int p=0;
        for (std::size_t i=0, size=json_array_size(cones); i<size; ++i) {
            const json_t* c=json_array_get(cones,i);
            auto ty=json_string_value(json_object_get(c,"type"));
            require(ty!=nullptr);
            std::string_view k=ty;
            int d=json_integer_value(json_object_get(c,"dim"));
            require(d>0);
            Type t=k=="zero"?Zero:k=="nonnegative"?Pos:k=="soc"?Soc:k=="exponential"?Exp:Psd;
            if (t==Psd && d==1) t=Pos;
            int64_t width=t==Psd ? int64_t(d)*(d+1)/2 : d;
            require(width <= m-p);
            K.push_back({t,d,p,p+static_cast<int>(width)});
            linear &= t==Zero || t==Pos;
            p+=width;
        }
        require(p==m);
    }
};

struct Result {
    int status=0; // none, solved, primal infeasible, dual infeasible
    V x,s,z;
};

double vmax(const V& x) {
    double n=0.;
    for (double z:x) {
        if (!std::isfinite(z)) return inf;
        n=std::max(n,std::abs(z));
    }
    return n;
}

double dot(const V& x,const V& y) {
    double t=0.;
    for (std::size_t j=0;j<x.size();++j) t+=x[j]*y[j];
    return t;
}

bool verify_linear(const Problem& p, Result& r) {
    const auto& [status,x,s,z]=r;
    if (status==1) {
        auto ax=p.A.mv(x), px=p.P.mv(x), az=p.A.tmv(z);
        if (!(std::isfinite(vmax(px)+vmax(ax)+vmax(az)+vmax(z)+vmax(s)))) return false;
        double dn=0.,pr=0.;
        for (int j=0;j<p.n;++j) dn=std::max(dn,std::abs(px[j]+p.q[j]+az[j]));
        if (dn>eps*(1.+std::max({vmax(p.q),vmax(px),vmax(az)}))) return false;
        for (int i=0;i<p.m;++i) pr=std::max(pr,std::abs(ax[i]+s[i]-p.b[i]));
        if (pr>eps*(1.+std::max({vmax(p.b),vmax(s),vmax(ax)}))) return false;
        double xx=dot(x,px)*.5;
        double y=xx+dot(x,p.q), w=-xx-dot(z,p.b);
        if (!std::isfinite(y+p.offset) || !std::isfinite(w+p.offset) || !std::isfinite(dot(s,z))
            || std::abs(y-w)>eps*std::max({1.,std::abs(y),std::abs(w)})) return false;
        for (auto c:p.K) {
            double sn=1.,zn=1.,sb=0.,zb=0.;
            for (int i=c.start;i<c.end;++i) {
                sn=std::max(sn,std::abs(s[i])); zn=std::max(zn,std::abs(z[i]));
                sb=std::max(sb,c.kind==Zero?std::abs(s[i]):-s[i]);
                if (c.kind==Pos) zb=std::max(zb,-z[i]);
            }
            if (sb>eps*sn || zb>eps*zn) return false;
        }
        return true;
    }
    if (status==2) {
        double nz=vmax(z), nb=vmax(p.b), na=vmax(p.A.v);
        if (!nz || !nb || !std::isfinite(nz)) return false;
        if (!na) na=1.;
        V y(p.m);
        double beta=0.;
        for (int i=0;i<p.m;++i) {
            y[i]=z[i]/nz;
            beta-=(p.b[i]/nb)*y[i];
        }
        if (!(beta>1.01e-8) || !std::isfinite(beta)) return false;
        for (int j=0;j<p.n;++j) {
            double v=0.;
            for (int t=p.A.p[j];t<p.A.p[j+1];++t) v+=p.A.v[t]/na*y[p.A.i[t]];
            if (std::abs(v)>eps*beta) return false;
        }
        for (auto c:p.K) if (c.kind==Pos)
            for (int i=c.start;i<c.end;++i) if (y[i]<-eps*beta) return false;
        return true;
    }
    if (status==3) {
        double nx=vmax(x), nq=vmax(p.q), na=vmax(p.A.v), nh=vmax(p.P.v);
        if (!nx || !nq || !std::isfinite(nx)) return false;
        if (!na) na=1.;
        if (!nh) nh=1.;
        V av(p.m),pv(p.n);
        double alpha=0.;
        for (int j=0;j<p.n;++j) {
            double vx=x[j]/nx;
            alpha-=(p.q[j]/nq)*vx;
            for (int t=p.A.p[j];t<p.A.p[j+1];++t) av[p.A.i[t]]+=(p.A.v[t]/na)*vx;
            for (int t=p.P.p[j];t<p.P.p[j+1];++t) pv[p.P.i[t]]+=(p.P.v[t]/nh)*vx;
        }
        if (!(alpha>1.01e-8) || !std::isfinite(alpha) || vmax(pv)>eps*alpha) return false;
        for (auto c:p.K) {
            double sn=1.,sb=0.;
            for (int i=c.start;i<c.end;++i) {
                const double sc=-av[i];
                sn=std::max(sn,std::abs(sc));
                sb=std::max(sb,c.kind==Zero?std::abs(sc):-sc);
            }
            if (sb>eps*alpha*sn) return false;
        }
        r.s.resize(p.m);
        for (int i=0;i<p.m;++i) r.s[i]=-av[i]*na*nx;
        return std::isfinite(vmax(r.s));
    }
    return false;
}

Result boxes(const Problem& p,Time stop) {
    if (!p.linear || p.A.v.size()>std::size_t(p.m)) return {};
    const int n=p.n,m=p.m;
    I witness(m,-1);
    std::vector<unsigned char> eq(m);
    for (auto c:p.K) if (c.kind==Zero)
        std::fill(eq.begin()+c.start,eq.begin()+c.end,1);
    V lo(n,-inf),hi(n,inf);
    I il(n,-1),iu(n,-1),cl(n,-1),cu(n,-1);
    for (int j=0;j<n;++j) for (int t=p.A.p[j];t<p.A.p[j+1];++t) {
        int i=p.A.i[t];
        if (witness[i]>=0) return {};
        witness[i]=t;
        double a=p.A.v[t], bound=p.b[i]/a;
        if (!std::isfinite(bound)) return {};
        if (a<0. || eq[i]) {
            if (bound>lo[j] || (bound==lo[j] && std::abs(a)>std::abs(p.A.v[il[j]]))) {
                lo[j]=bound; il[j]=t; cl[j]=i;
            }
        }
        if (a>0. || eq[i]) {
            if (bound<hi[j] || (bound==hi[j] && std::abs(a)>std::abs(p.A.v[iu[j]]))) {
                hi[j]=bound; iu[j]=t; cu[j]=i;
            }
        }
    }
    int bad=-1;
    for (int i=0;i<m;++i)
        if (witness[i]<0 && (p.b[i]<0. || (eq[i] && p.b[i]!=0.))
            && (bad<0 || std::abs(p.b[i])>std::abs(p.b[bad]))) bad=i;
    if (bad>=0) {
        Result r{2,{},{},V(m)};
        r.z[bad]=p.b[bad]<0.?1.:-1.;
        if (verify_linear(p,r)) return r;
        return {};
    }
    for (int j=0;j<n;++j) if (lo[j]>hi[j]) {
        double a=p.A.v[il[j]], b=p.A.v[iu[j]], d=std::min(std::abs(a),std::abs(b));
        Result r{2,{},{},V(m)};
        r.z[cl[j]]=-d/a;
        r.z[cu[j]]=d/b;
        if (verify_linear(p,r)) return r;
        return {};
    }
    V diag(n), x(n), ray(n);
    bool diagonal=true, unbounded=false;
    for (int j=0;j<n;++j) {
        for (int t=p.P.p[j];t<p.P.p[j+1];++t) {
            if (p.P.i[t]==j) diag[j]=p.P.v[t];
            else diagonal=false;
        }
        if (diag[j]<0. || (!diag[j] && p.P.p[j]!=p.P.p[j+1])) return {};
        double y=0.;
        if (diag[j]>0.) y=-p.q[j]/diag[j];
        else if (p.q[j]!=0.) {
            y=p.q[j]>0.?lo[j]:hi[j];
            if (!std::isfinite(y)) {
                ray[j]=p.q[j]>0.?-1.:1.;
                unbounded=true;
                y=0.;
            }
        }
        x[j]=std::clamp(y,lo[j],hi[j]);
    }
    if (unbounded) {
        Result r{3,std::move(ray),{},{}};
        if (verify_linear(p,r)) return r;
    }
    auto result=[&](const V& g) {
        Result r{1,x,p.A.mv(x),V(m)};
        for (int i=0;i<m;++i) r.s[i]=p.b[i]-r.s[i];
        for (int j=0;j<n;++j) {
            if (g[j]>0. && x[j]==lo[j] && il[j]>=0) r.z[cl[j]]=-g[j]/p.A.v[il[j]];
            else if (g[j]<0. && x[j]==hi[j] && iu[j]>=0) r.z[cu[j]]=-g[j]/p.A.v[iu[j]];
        }
        return r;
    };
    V g=p.P.mv(x);
    for (int j=0;j<n;++j) g[j]+=p.q[j];
    if (diagonal) {
        Result r=result(g);
        return verify_linear(p,r) ? std::move(r) : Result{};
    }
    if (p.P.v.size()>16*std::size_t(n)) return {};
    for (int it=0;it<46 && Clock::now()<stop;++it) {
        double step=0.;
        for (int j=0;j<n;++j) if (diag[j]>0.) {
            double xx=std::clamp(x[j]-g[j]/diag[j],lo[j],hi[j]);
            double dx=xx-x[j];
            if (dx!=0.) {
                x[j]=xx;
                step=std::max(step,std::abs(dx));
                for (int t=p.P.p[j];t<p.P.p[j+1];++t)
                    g[p.P.i[t]]+=p.P.v[t]*dx;
            }
        }
        if ((it%4==3 && step<1.e-4) || it%12==11 || step<1.e-12) {
            Result r=result(g);
            if (verify_linear(p,r)) return r;
        }
    }
    return {};
}

void put_entry(archive* ar,const char* name,std::string desc,std::string shape,
               const void* data,std::size_t bytes) {
    std::unique_ptr<archive_entry,decltype(&archive_entry_free)> ent(archive_entry_new(),&archive_entry_free);
    std::string head="{'descr': '"+desc+"', 'fortran_order': False, 'shape': "+shape+", }";
    head.append((63-(10+head.size())%64)%64,' ');
    head+='\n';
    std::array<char,10> magic={char(0x93),'N','U','M','P','Y',1,0,
                              char(head.size()%256),char(head.size()/256)};
    archive_entry_set_pathname(ent.get(),name);
    archive_entry_set_filetype(ent.get(),0100000);
    archive_entry_set_perm(ent.get(),0644);
    archive_entry_set_size(ent.get(),10+head.size()+bytes);
    require(archive_write_header(ar,ent.get())>=0);
    require(archive_write_data(ar,magic.data(),magic.size())==static_cast<long>(magic.size()));
    require(archive_write_data(ar,head.data(),head.size())==static_cast<long>(head.size()));
    if (bytes) require(archive_write_data(ar,data,bytes)==static_cast<long>(bytes));
}

void write_npz(const char* path,const Result& r) {
    std::unique_ptr<archive,decltype(&archive_write_free)> ar(archive_write_new(),&archive_write_free);
    require(bool(ar));
    require(archive_write_set_format_zip(ar.get())==0);
    require(archive_write_set_format_option(ar.get(),"zip","compression","store")==0);
    archive_write_set_bytes_per_block(ar.get(),0);
    require(archive_write_open_filename(ar.get(),path)==0);
    const char* status=r.status==1?"solved":r.status==2?"primal_infeasible":"dual_infeasible";
    std::string meta=std::string("{\"schema_version\":1,\"status\":\"")+status+"\"}";
    std::vector<uint32_t> text(meta.begin(),meta.end());
    put_entry(ar.get(),"metadata_json.npy","<U"+std::to_string(text.size()),"()",
              text.data(),4*text.size());
    auto vec=[&](const char* name,const V& v) {
        put_entry(ar.get(),name,"<f8","("+std::to_string(v.size())+",)",v.data(),8*v.size());
    };
    if (r.status!=2) { vec("x.npy",r.x); vec("s.npy",r.s); }
    if (r.status!=3) vec("z.npy",r.z);
    require(archive_write_close(ar.get())==0);
}

}

int main(int argc,char** argv) {
    auto start=Clock::now();
    const char* input=nullptr;
    const char* output=nullptr;
    for (int i=2;i+1<argc;++i) {
        std::string_view a=argv[i];
        if (a=="--input") input=argv[++i];
        else if (a=="--output") output=argv[++i];
    }
    if (!std::getenv("CONIC_PY_ONLY") && input && output) try {
        auto arrays=read_npz(input);
        Problem problem(arrays);
        arrays.clear();
        auto r=boxes(problem,start+std::chrono::milliseconds(380));
        if (r.status) {
            write_npz(output,r);
            return 0;
        }
    } catch (...) {
        // The Python path also handles unusual NPY encodings and input variants.
    }
    if (argc>1) {
        static char python[]="python";
        argv[0]=python;
        auto dt=std::chrono::duration<double>(Clock::now()-start).count();
        auto elapsed=std::to_string(dt);
        setenv("CONIC_PRELUDE_SECONDS",elapsed.c_str(),1);
        execvp(python,argv);
    }
    return 2;
}
