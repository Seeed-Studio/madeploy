#include "ma_transport_rtsp.h"


namespace ma {

static const char* TAG = "ma::transport::rtsp";

std::unordered_map<int, CVI_RTSP_CTX*> TransportRTSP::s_contexts;
std::unordered_map<int, std::vector<CVI_RTSP_SESSION*>> TransportRTSP::s_sessions;

TransportRTSP::TransportRTSP()
    : Transport(MA_TRANSPORT_RTSP),
      m_opened(false),
      m_format(MA_PIXEL_FORMAT_H264),
      m_port(0),
      m_session(nullptr),
      m_ctx(nullptr) {}

TransportRTSP::~TransportRTSP() {
    close();
}


static void onConnectStub(const char* ip, void* arg) {
    MA_LOGD(TAG, "rtsp connected: %s", ip);
}

static void onDisconnectStub(const char* ip, void* arg) {
    MA_LOGD(TAG, "rtsp disconnected: %s", ip);
}
ma_err_t TransportRTSP::open(int port, const std::string name, ma_pixel_format_t format) {
    ma_err_t ret;
    Guard guard(m_mutex);

    // only support H264
    if (m_format != MA_PIXEL_FORMAT_H264) {
        return MA_ENOTSUP;
    }

    if (m_opened) {
        return MA_EBUSY;
    }

    if (s_contexts.find(port) == s_contexts.end()) {
        CVI_RTSP_CONFIG config = {0};
        config.port            = port;
        if (CVI_RTSP_Create(&m_ctx, &config) < 0) {
            return MA_EINVAL;
        }
        s_contexts[port] = m_ctx;
        if (CVI_RTSP_Start(m_ctx) < 0) {
            CVI_RTSP_Destroy(&m_ctx);
            return MA_EINVAL;
        }
        CVI_RTSP_STATE_LISTENER listener = {0};
        listener.onConnect               = onConnectStub;
        listener.argConn                 = m_ctx;
        listener.onDisconnect            = onDisconnectStub;
        listener.argDisconn              = m_ctx;

        CVI_RTSP_SetListener(m_ctx, &listener);

        MA_LOGV(TAG, "rtsp sever created: %d", port);

    } else {
        m_ctx = s_contexts[port];
    }

    for (auto& session : s_sessions[port]) {
        if (session->name == name) {
            MA_LOGW(TAG, "rtsp session already exists: %d/%s", port, name.c_str());
            return MA_EBUSY;
        }
    }
    CVI_RTSP_SESSION_ATTR attr = {0};

    attr.reuseFirstSource = true;

    switch (format) {
        case MA_PIXEL_FORMAT_H264:
            attr.video.codec = RTSP_VIDEO_H264;
            break;
        case MA_PIXEL_FORMAT_H265:
            attr.video.codec = RTSP_VIDEO_H265;
            break;
        case MA_PIXEL_FORMAT_JPEG:
            attr.video.codec = RTSP_VIDEO_JPEG;
            break;
        default:
            return MA_EINVAL;
    }

    snprintf(attr.name, sizeof(attr.name), "%s", name.c_str());

    CVI_RTSP_CreateSession(m_ctx, &attr, &m_session);

    s_sessions[port].push_back(m_session);


    MA_LOGI(TAG, "rtsp session created: %d/%s", port, name.c_str());

    m_port   = port;
    m_name   = name;
    m_opened = true;
    m_format = format;

    return MA_OK;
}


ma_err_t TransportRTSP::close() {
    Guard guard(m_mutex);
    if (!m_opened) {
        return MA_EINVAL;
    }
    CVI_RTSP_DestroySession(m_ctx, m_session);

    MA_LOGI(TAG, "rtsp session destroyed: %d/%s", m_port, m_name.c_str());

    auto it = std::find(s_sessions[m_port].begin(), s_sessions[m_port].end(), m_session);
    if (it != s_sessions[m_port].end()) {
        s_sessions[m_port].erase(it);
    }

    if (s_sessions[m_port].empty()) {
        MA_LOGV(TAG, "rtsp server destroy: %d", m_port);
        CVI_RTSP_Stop(m_ctx);
        CVI_RTSP_Destroy(&m_ctx);
        s_contexts.erase(m_port);
    }

    m_opened = false;
    return MA_OK;
}


TransportRTSP::operator bool() const {
    return m_opened;
}


size_t TransportRTSP::available() const {
    return 0;
}


size_t TransportRTSP::send(const char* data, size_t length, int timeout) {
    Guard guard(m_mutex);
    if (!m_opened) {
        return 0;
    }
    CVI_RTSP_DATA frame = {0};
    frame.blockCnt      = 1;
    frame.dataPtr[0]    = reinterpret_cast<uint8_t*>(const_cast<char*>(data));
    frame.dataLen[0]    = length;

    CVI_RTSP_WriteFrame(s_contexts[m_port], m_session->video, (CVI_RTSP_DATA*)&frame);

    return 0;
}


size_t TransportRTSP::receive(char* data, size_t length, int timeout) {
    return 0;
}


size_t TransportRTSP::receiveUtil(char* data, size_t length, char delimiter, int timeout) {
    return 0;
}


}  // namespace ma
