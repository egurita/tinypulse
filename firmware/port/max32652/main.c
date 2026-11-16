/* Emulator benchmark firmware for the MAX32652.
 *
 * Replays an embedded real-data segment through the full chain:
 *   dataset sample -> mock MAX30102 (registers + FIFO) -> I2C HAL -> driver
 *   -> streaming estimator -> UART ("HR <window> <bin> <milli-BPM>")
 * bench_mark(id) calls are hooked by Renode, which logs the executed-instruction
 * counter, giving exact per-call instruction counts.
 */
#include "board.h"
#include "hr_estimator.h"
#include "max30102.h"
#include "max30102_mock.h"
#include "test_vector.h"

enum { MARK_PUSH_START = 1, MARK_PUSH_END = 2, MARK_WINDOW_END = 3,
       MARK_FIFO_START = 4, MARK_FIFO_END = 5, MARK_DONE = 9 };

#define FIFO_READ_EVERY 16      /* samples between driver reads (A_FULL would fire at 17) */

static hr_estimator_t est;
static max30102_mock_t sensor;
static max30102_t dev;
static max30102_sample_t fifo_buf[MAX30102_FIFO_DEPTH];

static const max30102_config_t SENSOR_CFG = {
    /* A real MAX30102 cannot sample at 64 Hz (50/100/200... sps). The dataset was
     * recorded at 64 Hz, so the mock is clocked at 64 Hz; on hardware the pipeline
     * would run at 100 sps or the samples would be resampled. */
    .sample_rate = MAX30102_SR_100, .pulse_width = MAX30102_PW_411US_18BIT,
    .adc_range = MAX30102_RANGE_4096NA, .sample_avg_log2 = 0,
    .led_red_pa = 0x24, .led_ir_pa = 0x24, .fifo_a_full = 15,
};

/* One pass over the test vector. DSP results are printed as "HR ...", NN results as
 * "HN ..."; NN markers are offset by 10 so the two can be told apart in the log. */
static void run(hr_mode_t mode, const uint16_t *expected, uint32_t mark_base) {
    mock_max30102_init(&sensor);
    hal_i2c_t hal = mock_max30102_hal(&sensor);
    int st = max30102_init(&dev, &hal, &SENSOR_CFG);
    uart_puts("SENSOR init=");
    uart_puti(st);
    uart_putc('\n');
    hr_init(&est);
    hr_set_mode(&est, mode);

    uint32_t fed = 0, mismatches = 0, n_out = 0;
    for (uint32_t i = 0; i < TV_N_PPG; i++) {
        mock_max30102_push(&sensor, tv_ppg[i], tv_ppg[i]);   /* IR channel carries the PPG */
        if ((i + 1) % FIFO_READ_EVERY != 0 && i + 1 != TV_N_PPG) continue;

        bench_mark(mark_base + MARK_FIFO_START);
        int n = max30102_read_fifo(&dev, fifo_buf, MAX30102_FIFO_DEPTH);
        bench_mark(mark_base + MARK_FIFO_END);
        for (int j = 0; j < n; j++, fed++) {
            const int16_t *acc = (fed % 2 == 0) ? &tv_acc[3 * (fed / 2)] : 0;
            hr_result_t r;
            bench_mark(mark_base + MARK_PUSH_START);
            int ready = hr_push(&est, fifo_buf[j].ir, acc, &r);
            bench_mark(mark_base + (ready ? MARK_WINDOW_END : MARK_PUSH_END));
            if (ready) {
                uart_puts(mode == HR_MODE_NN ? "HN " : "HR ");
                uart_putu(r.window_index); uart_putc(' ');
                uart_putu(r.bin); uart_putc(' ');
                uart_putu(r.bpm_x1000); uart_putc('\n');
                if (n_out >= TV_N_EXPECTED || r.bin != expected[n_out]) mismatches++;
                n_out++;
            }
        }
    }
    uart_puts(mode == HR_MODE_NN ? "DONE mode=nn" : "DONE mode=dsp");
    uart_puts(" samples="); uart_putu(fed);
    uart_puts(" estimates="); uart_putu(n_out);
    uart_puts(" mismatches_vs_host="); uart_putu(mismatches);
    uart_puts(" samples_lost="); uart_putu(dev.samples_lost);
    uart_puts(" stack_used="); uart_putu(stack_high_water());
    uart_putc('\n');
}

int main(void) {
    uart_init();
    uart_puts("BOOT hr-firmware max32652\n");
    run(HR_MODE_DSP, tv_expected_bins, 0);
    run(HR_MODE_NN, tv_expected_bins_nn, 10);
    bench_mark(MARK_DONE);
    /* Sleep instead of spinning: on hardware this is the low-power idle, and in the
     * emulator the rest of the simulated time then costs nothing to emulate. */
    for (;;) __asm volatile("wfi");
}
