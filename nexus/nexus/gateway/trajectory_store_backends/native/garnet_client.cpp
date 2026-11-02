#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "hiredis.h"
}

namespace py = pybind11;

namespace {

struct ReplyDeleter {
    void operator()(redisReply* reply) const { freeReplyObject(reply); }
};

using Reply = std::unique_ptr<redisReply, ReplyDeleter>;

// NOTE(jerryhrzhou): Each exported buffer owns one reply, independent of both the connection and other fields.
class NativeBuffer {
public:
    explicit NativeBuffer(Reply reply) : reply_(std::move(reply)) {}

    py::buffer_info buffer() {
        return py::buffer_info(
            reply_->str, 1, py::format_descriptor<unsigned char>::format(), 1,
            {static_cast<py::ssize_t>(reply_->len)}, {1});
    }

private:
    Reply reply_;
};

class NativeGarnetClient {
public:
    NativeGarnetClient(const std::string& host, int port) {
        const timeval connect_timeout{5, 0};
        context_.reset(redisConnectWithTimeout(host.c_str(), port, connect_timeout));
        if (!context_) {
            throw std::bad_alloc();
        }
        if (context_->err) {
            throw std::runtime_error("Garnet connect failed: " + std::string(context_->errstr));
        }
        const timeval command_timeout{120, 0};
        if (redisSetTimeout(context_.get(), command_timeout) != REDIS_OK) {
            throw std::runtime_error("Garnet timeout setup failed: " + std::string(context_->errstr));
        }
    }

    std::vector<std::shared_ptr<NativeBuffer>> mget(const std::vector<std::string>& keys) {
        if (keys.empty()) {
            return {};
        }
        if (!context_) {
            throw std::runtime_error("Garnet connection is closed");
        }
        if (keys.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error("Too many keys for hiredis MGET");
        }
        std::vector<const char*> arguments;
        std::vector<size_t> lengths;
        arguments.reserve(keys.size() + 1);
        lengths.reserve(keys.size() + 1);
        arguments.push_back("MGET");
        lengths.push_back(4);
        for (const auto& key : keys) {
            arguments.push_back(key.data());
            lengths.push_back(key.size());
        }

        Reply response(static_cast<redisReply*>(redisCommandArgv(
            context_.get(), static_cast<int>(arguments.size()), arguments.data(), lengths.data())));
        if (!response) {
            throw std::runtime_error("Garnet MGET failed: " + std::string(context_->errstr));
        }
        if (response->type == REDIS_REPLY_ERROR) {
            throw std::runtime_error("Garnet MGET error: " + std::string(response->str, response->len));
        }
        if (response->type != REDIS_REPLY_ARRAY || response->elements != keys.size()) {
            throw std::runtime_error("Garnet MGET returned an invalid response array");
        }

        std::vector<std::shared_ptr<NativeBuffer>> buffers;
        buffers.reserve(keys.size());
        for (size_t index = 0; index < response->elements; ++index) {
            auto* element = response->element[index];
            if (!element) {
                throw std::runtime_error("Garnet MGET returned a null reply element");
            }
            if (element->type == REDIS_REPLY_NIL) {
                buffers.push_back(nullptr);
                continue;
            }
            if (element->type != REDIS_REPLY_STRING ||
                element->len > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max())) {
                throw std::runtime_error("Garnet MGET returned an invalid bulk string");
            }
            Reply field(element);
            response->element[index] = nullptr;
            buffers.push_back(std::make_shared<NativeBuffer>(std::move(field)));
        }
        return buffers;
    }

    void close() { context_.reset(); }

private:
    std::unique_ptr<redisContext, decltype(&redisFree)> context_{nullptr, redisFree};
};

}  // namespace

PYBIND11_MODULE(_garnet_native, module) {
    module.doc() = "Single-connection Garnet MGET with GIL-free native I/O and owned response buffers.";
    module.attr("hiredis_version") = "1.3.0";
    py::class_<NativeBuffer, std::shared_ptr<NativeBuffer>>(module, "NativeBuffer", py::buffer_protocol())
        .def_buffer(&NativeBuffer::buffer);
    py::class_<NativeGarnetClient>(module, "NativeGarnetClient")
        .def(py::init([](const std::string& host, int port) {
            py::gil_scoped_release release;
            return std::make_unique<NativeGarnetClient>(host, port);
        }), py::arg("host"), py::arg("port"))
        .def("mget", &NativeGarnetClient::mget, py::arg("keys"), py::call_guard<py::gil_scoped_release>())
        .def("close", &NativeGarnetClient::close, py::call_guard<py::gil_scoped_release>());
}
