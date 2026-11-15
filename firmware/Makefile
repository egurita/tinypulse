# Heart-rate firmware build.
#   make native    host shared library (for Python/ctypes) + C unit tests
#   make test      build and run the C unit tests
#   make arm       Cortex-M4 image for the MAX32652 (runs in Renode)
#   make size      flash/RAM usage of the ARM image, per module
#   make tables    regenerate gen/hr_tables.[ch] from the Python reference

BUILD   := build
CORE    := src/bandpass.c src/nlms.c src/spectrum.c src/track.c src/hr_estimator.c \
           src/max30102.c gen/hr_tables.c
INC     := -Iinclude -Igen -Imock
WARN    := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -std=c99

# ---------------- native (macOS/Linux host) ----------------
CC      ?= cc
HOST_CFLAGS := $(WARN) $(INC) -O2 -g -fPIC
SHLIB_EXT := $(if $(filter Darwin,$(shell uname)),dylib,so)
LIBHR   := $(BUILD)/native/libhr.$(SHLIB_EXT)
TESTBIN := $(BUILD)/native/test_firmware
# AddressSanitizer hangs at startup on this macOS release (even for an empty program),
# so macOS uses UBSan only; Linux (CI) gets ASan + UBSan.
SANITIZE := $(if $(filter Darwin,$(shell uname)),undefined,address,undefined)
TESTS   := tests/test_main.c tests/test_dsp.c tests/test_max30102.c mock/max30102_mock.c

native: $(LIBHR) $(TESTBIN)

$(LIBHR): $(CORE) host/hr_host.c mock/max30102_mock.c $(wildcard include/*.h gen/*.h mock/*.h)
	@mkdir -p $(dir $@)
	$(CC) $(HOST_CFLAGS) -shared -o $@ $(CORE) host/hr_host.c mock/max30102_mock.c

$(TESTBIN): $(CORE) $(TESTS) $(wildcard include/*.h gen/*.h mock/*.h tests/*.h)
	@mkdir -p $(dir $@)
	$(CC) $(HOST_CFLAGS) -fsanitize=$(SANITIZE) -fno-sanitize-recover=undefined -o $@ $(CORE) $(TESTS) -lm

test: $(TESTBIN)
	./$(TESTBIN)

# ---------------- ARM Cortex-M4 (MAX32652) ----------------
ARM_CC  := arm-none-eabi-gcc
ARM_DIR := $(BUILD)/arm
ARCH    := -mcpu=cortex-m4 -mthumb -mfloat-abi=soft
ARM_CFLAGS := $(WARN) $(INC) -Iport/max32652 $(ARCH) -O2 -g -ffreestanding \
              -ffunction-sections -fdata-sections -fno-common
ARM_LDFLAGS := $(ARCH) -nostdlib -nostartfiles -T port/max32652/max32652.ld \
               -Wl,--gc-sections -Wl,-Map=$(ARM_DIR)/fw.map -Wl,--print-memory-usage
PORT    := port/max32652/startup.c port/max32652/uart.c port/max32652/mini_libc.c \
           port/max32652/main.c mock/max30102_mock.c gen/test_vector.c
ARM_ELF := $(ARM_DIR)/fw.elf

arm: $(ARM_ELF)

$(ARM_ELF): $(CORE) $(PORT) $(wildcard include/*.h gen/*.h mock/*.h port/max32652/*)
	@mkdir -p $(ARM_DIR)
	$(ARM_CC) $(ARM_CFLAGS) -o $@ $(CORE) $(PORT) $(ARM_LDFLAGS) -lgcc

size: $(ARM_ELF)
	arm-none-eabi-size -A $(ARM_ELF) | grep -E "^\.(text|rodata|data|bss|stack)"
	arm-none-eabi-nm --size-sort -S -C $(ARM_ELF) | tail -25

tables:
	cd .. && uv run python python/scripts/gen_firmware_tables.py

clean:
	rm -rf $(BUILD)

.PHONY: native test arm size tables clean
