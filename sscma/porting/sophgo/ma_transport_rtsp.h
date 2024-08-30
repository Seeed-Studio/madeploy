#ifndef _MA_TRANSPORT_RTSP_H
#define _MA_TRANSPORT_RTSP_H

#include <sscma.h>

#include <rtsp.h>

namespace ma {

class TransportRTSP final : public Transport {

public:
    TransportRTSP();
    ~TransportRTSP();

    ma_err_t open(int port,
                  const std::string name   = "live",
                  ma_pixel_format_t format = MA_PIXEL_FORMAT_H264);
    ma_err_t close();

    operator bool() const override;


    size_t available() const override;
    size_t send(const char* data, size_t length, int timeout = -1) override;
    size_t receive(char* data, size_t length, int timeout = 1) override;
    size_t receiveUtil(char* data, size_t length, char delimiter, int timeout = 1) override;

private:
    ma_pixel_format_t m_format;
    int m_port;
    std::string m_name;
    std::atomic<bool> m_opened;
    CVI_RTSP_SESSION* m_session;
    CVI_RTSP_CTX* m_ctx;
    Mutex m_mutex;
    std::vector<std::string> m_ip_list;
    static std::unordered_map<int, CVI_RTSP_CTX*> s_contexts;
    static std::unordered_map<int, std::vector<CVI_RTSP_SESSION*>> s_sessions;
};

}  // namespace ma


#endif