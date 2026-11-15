/* Software-in-the-loop application (see sil_app.h for the protocol). Portable:
 * used by the native host build and by the MAX32652 image. */
#include "sil_app.h"
#include "hr_estimator.h"
#include "max30102.h"
#include "max30102_mock.h"

#define POLL_EVERY 4        /* read the sensor FIFO every 4 samples (62.5 ms) */
#define ACC_QUEUE  64       /* ACC rides with each frame; queued until its PPG sample is read */

static sil_write_fn out;
static hr_estimator_t est;
static max30102_mock_t sensor;
static hal_i2c_t hal;
static max30102_t dev;
static max30102_sample_t fifo[MAX30102_FIFO_DEPTH];

static uint8_t frame[SIL_FRAME_LEN];
static uint8_t flen, expected_seq, have_seq, done;
static uint32_t rx_frames, chk_errors, seq_errors, n_est, since_poll;

static struct { uint8_t has; int16_t xyz[3]; } accq[ACC_QUEUE];
static uint32_t acc_head, acc_tail;

/* ---- tiny formatting (no printf on the MCU) ---- */
static char *put_u(char *p, uint32_t v) {
    char tmp[10];
    int n = 0;
    do { tmp[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) *p++ = tmp[--n];
    return p;
}
static char *put_s(char *p, const char *s) { while (*s) *p++ = *s++; return p; }

static void emit_kv(const char *head, const char *const *keys, const uint32_t *vals, int n) {
    char line[128], *p = put_s(line, head);
    for (int i = 0; i < n; i++) { *p++ = ' '; p = put_s(p, keys[i]); p = put_u(p, vals[i]); }
    *p++ = '\n'; *p = 0;
    out(line);
}

static void drain_sensor(void) {
    int n = max30102_read_fifo(&dev, fifo, MAX30102_FIFO_DEPTH);
    for (int i = 0; i < n; i++) {
        const int16_t *acc = 0;
        if (acc_tail != acc_head) {
            if (accq[acc_tail % ACC_QUEUE].has) acc = accq[acc_tail % ACC_QUEUE].xyz;
            acc_tail++;
        }
        hr_result_t r;
        if (hr_push(&est, fifo[i].ir, acc, &r)) {
            char line[48], *p = put_s(line, "HR ");
            p = put_u(p, r.window_index); *p++ = ' ';
            p = put_u(p, r.bin); *p++ = ' ';
            p = put_u(p, r.bpm_x1000); *p++ = '\n'; *p = 0;
            out(line);
            n_est++;
        }
    }
}

static void handle_frame(void) {
    uint8_t seq = frame[1], flags = frame[2];
    rx_frames++;
    if (have_seq && seq != expected_seq) {
        const char *k[] = {"", ""};
        uint32_t v[] = {expected_seq, seq};
        emit_kv("ERR SEQ", k, v, 2);
        seq_errors++;
    }
    expected_seq = (uint8_t)(seq + 1);
    have_seq = 1;

    if (flags & SIL_FLAG_END) {
        drain_sensor();
        const char *k[] = {"rx=", "chk=", "seq=", "lost=", "est="};
        uint32_t v[] = {rx_frames, chk_errors, seq_errors, dev.samples_lost, n_est};
        emit_kv("DONE", k, v, 5);
        done = 1;
        return;
    }
    uint32_t ppg = (uint32_t)frame[3] | ((uint32_t)frame[4] << 8) | ((uint32_t)frame[5] << 16);
    int stored_before = mock_max30102_unread(&sensor);
    mock_max30102_push(&sensor, ppg, ppg);
    if (mock_max30102_unread(&sensor) != stored_before) {          /* not lost to overflow */
        uint32_t slot = acc_head++ % ACC_QUEUE;
        accq[slot].has = (flags & SIL_FLAG_ACC) ? 1 : 0;
        for (int c = 0; c < 3; c++)
            accq[slot].xyz[c] = (int16_t)(frame[6 + 2 * c] | (frame[7 + 2 * c] << 8));
    }
    if (++since_poll >= POLL_EVERY) { since_poll = 0; drain_sensor(); }
}

void sil_init(sil_write_fn write_line) {
    static const max30102_config_t cfg = {
        .sample_rate = MAX30102_SR_100, .pulse_width = MAX30102_PW_411US_18BIT,
        .adc_range = MAX30102_RANGE_4096NA, .sample_avg_log2 = 0,
        .led_red_pa = 0x24, .led_ir_pa = 0x24, .fifo_a_full = 15,
    };
    out = write_line;
    mock_max30102_init(&sensor);
    hal = mock_max30102_hal(&sensor);
    hr_init(&est);
    flen = 0; have_seq = 0; done = 0;
    rx_frames = chk_errors = seq_errors = n_est = since_poll = 0;
    acc_head = acc_tail = 0;
    out(max30102_init(&dev, &hal, &cfg) == MAX30102_OK ? "READY\n" : "ERR SENSOR\n");
}

void sil_rx_byte(uint8_t b) {
    if (flen == 0 && b != SIL_SYNC) return;              /* hunt for sync */
    frame[flen++] = b;
    if (flen < SIL_FRAME_LEN) return;

    uint8_t x = 0;
    for (int i = 1; i < SIL_FRAME_LEN - 1; i++) x ^= frame[i];
    if (x == frame[SIL_FRAME_LEN - 1]) {
        flen = 0;
        handle_frame();
        return;
    }
    /* Bad checksum: report, then resync from the next 0xA5 inside the buffer. */
    const char *k[] = {""};
    uint32_t v[] = {frame[1]};
    emit_kv("ERR CHK", k, v, 1);
    chk_errors++;
    uint8_t i = 1;
    while (i < SIL_FRAME_LEN && frame[i] != SIL_SYNC) i++;
    uint8_t keep = (uint8_t)(SIL_FRAME_LEN - i);
    for (uint8_t j = 0; j < keep; j++) frame[j] = frame[i + j];
    flen = keep;
}

int sil_done(void) { return done; }
