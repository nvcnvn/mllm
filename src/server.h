// OpenAI-style server over an Engine: POST /v1/completions and /v1/chat/completions (messages through the chat
// template; the reasoning before "</think>" comes as reasoning_content), streamed (SSE) or not, with stop strings;
// GET /v1/models. One engine thread owns the Engine (architecture.md rule 3); httplib's handler threads only queue
// requests and read their text. A client that disconnects cancels its request.
//
//   mllm::Server s(engine, "mlx-community/Qwen3.5-0.8B-4bit");
//   s.listen("127.0.0.1", 8080);  // blocks until s.stop() from another thread
#pragma once
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "engine.h"

namespace httplib {
class Server;
struct Request;
struct Response;
}

namespace mllm {

class Server {
public:
    Server(Engine& e, std::string model_name);
    ~Server();
    bool listen(const std::string& host, int port);  // blocks until stop(); false if the port can't be bound
    void wait_until_ready() const;
    // Stops accepting and cuts the open streams (their requests are cancelled); listen() returns once the engine
    // thread is done.
    void stop();
    size_t live_requests();  // the engine's live requests, as of its last loop

    struct Stream;  // one request: its prompt in, its events out

private:
    void loop();  // the engine thread
    void handle(const httplib::Request& req, httplib::Response& res, bool chat);

    Engine& e;
    std::string model;
    std::unique_ptr<httplib::Server> http;
    std::mutex m;  // guards jobs, cancels, stopping, n_live
    std::condition_variable cv;
    std::deque<std::shared_ptr<Stream>> jobs, cancels;
    bool stopping = false;
    size_t n_live = 0;
};

}  // namespace mllm
