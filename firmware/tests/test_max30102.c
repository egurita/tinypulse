/* Driver tests against the register-level mock. */
#include "ctest.h"
#include "max30102.h"
#include "max30102_mock.h"

static max30102_mock_t mock;
static hal_i2c_t hal;
static max30102_t dev;

static const max30102_config_t CFG = {
    .sample_rate = MAX30102_SR_100, .pulse_width = MAX30102_PW_411US_18BIT,
    .adc_range = MAX30102_RANGE_4096NA, .sample_avg_log2 = 0,
    .led_red_pa = 0x24, .led_ir_pa = 0x24, .fifo_a_full = 15,
};

static void setup(void) {
    mock_max30102_init(&mock);
    hal = mock_max30102_hal(&mock);
}

static void test_init_configures_registers(void) {
    setup();
    CHECK_EQ(max30102_init(&dev, &hal, &CFG), MAX30102_OK);
    CHECK_EQ(mock.regs[MAX30102_REG_MODE_CONFIG], MAX30102_MODE_SPO2);
    CHECK_EQ(mock.regs[MAX30102_REG_SPO2_CONFIG], (1 << 5) | (1 << 2) | 3);
    CHECK_EQ(mock.regs[MAX30102_REG_FIFO_CONFIG], 0x0F);
    CHECK_EQ(mock.regs[MAX30102_REG_LED1_PA], 0x24);
    CHECK_EQ(mock.regs[MAX30102_REG_INTR_ENABLE_1], MAX30102_INT_A_FULL);
}

static void test_wrong_part_id_rejected(void) {
    setup();
    mock.part_id = 0x11;                      /* e.g. a MAX30100 at the same address */
    mock_max30102_init(&mock); mock.part_id = 0x11; mock.regs[MAX30102_REG_PART_ID] = 0x11;
    CHECK_EQ(max30102_init(&dev, &hal, &CFG), MAX30102_ERR_ID);
}

static void test_bus_nack_reported(void) {
    setup();
    mock.nack = 1;
    CHECK_EQ(max30102_init(&dev, &hal, &CFG), MAX30102_ERR_BUS);
}

static void test_reset_timeout(void) {
    setup();
    mock.stuck_reset = 1;
    CHECK_EQ(max30102_init(&dev, &hal, &CFG), MAX30102_ERR_TIMEOUT);
}

static void test_invalid_config_rejected_before_bus_traffic(void) {
    setup();
    max30102_config_t bad = CFG;
    bad.sample_rate = MAX30102_SR_1000;       /* 411 us pulses allow at most 400 sps */
    CHECK_EQ(max30102_init(&dev, &hal, &bad), MAX30102_ERR_CONFIG);
    CHECK_EQ(mock.transactions, 0);
}

static void test_samples_roundtrip_in_order(void) {
    setup();
    max30102_init(&dev, &hal, &CFG);
    for (uint32_t i = 0; i < 10; i++) mock_max30102_push(&mock, 1000 + i, 0x3FFFF - i);
    max30102_sample_t s[32];
    CHECK_EQ(max30102_read_fifo(&dev, s, 32), 10);
    int ok = 1;
    for (uint32_t i = 0; i < 10; i++) ok &= s[i].red == 1000 + i && s[i].ir == 0x3FFFF - i;
    CHECK(ok);
    CHECK_EQ(max30102_read_fifo(&dev, s, 32), 0);
}

static void test_pointer_wraparound(void) {
    setup();
    max30102_init(&dev, &hal, &CFG);
    max30102_sample_t s[32];
    uint32_t next = 0;
    int ok = 1;
    for (int round = 0; round < 10; round++) {          /* 10 x 25 samples: wraps many times */
        for (int i = 0; i < 25; i++) mock_max30102_push(&mock, next + (uint32_t)i, 0);
        int n = max30102_read_fifo(&dev, s, 32);
        ok &= n == 25;
        for (int i = 0; i < n; i++) ok &= s[i].red == next + (uint32_t)i;
        next += 25;
    }
    CHECK(ok);
}

static void test_partial_reads(void) {
    setup();
    max30102_init(&dev, &hal, &CFG);
    for (uint32_t i = 0; i < 20; i++) mock_max30102_push(&mock, i, i);
    max30102_sample_t s[8];
    CHECK_EQ(max30102_read_fifo(&dev, s, 8), 8);
    CHECK_EQ(s[7].red, 7);
    CHECK_EQ(max30102_read_fifo(&dev, s, 8), 8);
    CHECK_EQ(s[0].red, 8);
    CHECK_EQ(max30102_read_fifo(&dev, s, 8), 4);
}

static void test_overflow_counts_lost_samples(void) {
    setup();
    max30102_init(&dev, &hal, &CFG);
    for (uint32_t i = 0; i < 40; i++) mock_max30102_push(&mock, i, i);   /* 8 too many */
    max30102_sample_t s[32];
    CHECK_EQ(max30102_read_fifo(&dev, s, 32), 32);
    CHECK_EQ(s[0].red, 0);                    /* rollover off: the NEWEST samples are lost */
    CHECK_EQ(s[31].red, 31);
    CHECK_EQ(dev.samples_lost, 8);
}

static void test_exactly_full_fifo_looks_empty(void) {
    /* Datasheet quirk: 32 samples and no overflow yet -> WR_PTR == RD_PTR, OVF == 0,
     * indistinguishable from empty. This is why the driver enables A_FULL and must
     * read before the FIFO fills. The test documents the behaviour. */
    setup();
    max30102_init(&dev, &hal, &CFG);
    for (uint32_t i = 0; i < 32; i++) mock_max30102_push(&mock, i, i);
    max30102_sample_t s[32];
    CHECK_EQ(max30102_read_fifo(&dev, s, 32), 0);
    CHECK_EQ(mock_max30102_unread(&mock), 32);
}

static void test_a_full_interrupt_and_clear_on_read(void) {
    setup();
    max30102_init(&dev, &hal, &CFG);          /* A_FULL = 15 free slots -> fires at 17 */
    max30102_read_int_status(&dev);           /* clear PWR_RDY */
    for (uint32_t i = 0; i < 16; i++) mock_max30102_push(&mock, i, i);
    CHECK(!(max30102_read_int_status(&dev) & MAX30102_INT_A_FULL));
    mock_max30102_push(&mock, 16, 16);
    CHECK(max30102_read_int_status(&dev) & MAX30102_INT_A_FULL);
    CHECK_EQ(max30102_read_int_status(&dev) & MAX30102_INT_A_FULL, 0);
}

static void test_shutdown_stops_sampling(void) {
    setup();
    max30102_init(&dev, &hal, &CFG);
    CHECK_EQ(max30102_set_shutdown(&dev, 1), MAX30102_OK);
    mock_max30102_push(&mock, 1, 1);
    CHECK_EQ(mock_max30102_unread(&mock), 0);
    max30102_set_shutdown(&dev, 0);
    mock_max30102_push(&mock, 1, 1);
    CHECK_EQ(mock_max30102_unread(&mock), 1);
}

static void test_resolution_follows_pulse_width(void) {
    setup();
    max30102_config_t c = CFG;
    c.pulse_width = MAX30102_PW_69US_15BIT;
    max30102_init(&dev, &hal, &c);
    mock_max30102_push(&mock, 0x3FFFF, 0x12345);
    max30102_sample_t s;
    CHECK_EQ(max30102_read_fifo(&dev, &s, 1), 1);
    CHECK_EQ(s.red, 0x3FFF8);                 /* 15-bit: lowest 3 bits are zero */
    CHECK_EQ(s.ir, 0x12340);
}

void run_max30102_tests(void) {
    RUN(test_init_configures_registers);
    RUN(test_wrong_part_id_rejected);
    RUN(test_bus_nack_reported);
    RUN(test_reset_timeout);
    RUN(test_invalid_config_rejected_before_bus_traffic);
    RUN(test_samples_roundtrip_in_order);
    RUN(test_pointer_wraparound);
    RUN(test_partial_reads);
    RUN(test_overflow_counts_lost_samples);
    RUN(test_exactly_full_fifo_looks_empty);
    RUN(test_a_full_interrupt_and_clear_on_read);
    RUN(test_shutdown_stops_sampling);
    RUN(test_resolution_follows_pulse_width);
}
