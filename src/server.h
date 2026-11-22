// Minimal OpenAI-style server over an Engine, just what tools/serve_bench.py needs: POST /v1/completions with
// stream: true (SSE), GET /v1/models. One engine thread owns the Engine (architecture.md rule 3); httplib's handler
// threads only queue requests and read their events. A client that disconnects cancels its request.
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
