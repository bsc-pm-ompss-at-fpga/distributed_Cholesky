.PHONY: info

PROGRAM_     = cholesky

CC           = clang
CFLAGS_      = $(CFLAGS) -O3 -fompss-2 -Wall -Wno-unused -Wno-unknown-pragmas

# FPGA bitstream Variables
FPGA_CLOCK             ?= 200
FPGA_MEMORY_PORT_WIDTH ?= 128
SYRK_NUM_ACCS          ?= 1
GEMM_NUM_ACCS          ?= 1
TRSM_NUM_ACCS          ?= 1
BLOCK_SIZE             ?= 128
POTRF_SMP              ?= 0
FPGA_GEMM_II           ?= 1
FPGA_OTHER_II          ?= 1
FROM_STEP = HLS
TO_STEP = bitstream

ifeq ($(POTRF_SMP),1)
	CFLAGS_ += -DPOTRF_SMP
endif

## MKL Variables
MKL_DIR      ?= $(MKLROOT)
MKL_INC_DIR  ?= $(MKL_DIR)/include
MKL_LIB_DIR  ?= $(MKL_DIR)/lib
MKL_SUPPORT_ = $(if $(and $(wildcard $(MKL_INC_DIR)/mkl.h ), \
               $(wildcard $(MKL_LIB_DIR)/libmkl_sequential.so )),YES,NO)

## Open Blas Variables
OPENBLAS_DIR      ?= $(OPENBLAS_HOME)
OPENBLAS_INC_DIR  ?= $(OPENBLAS_DIR)/include
OPENBLAS_LIB_DIR  ?= $(OPENBLAS_DIR)/lib
OPENBLAS_SUPPORT_ = $(if $(and $(wildcard $(OPENBLAS_INC_DIR)/lapacke.h ), \
                    $(wildcard $(OPENBLAS_LIB_DIR)/libopenblas.so )),YES,NO)

ifeq ($(MKL_SUPPORT_),YES)
	CFLAGS_  += -I$(MKL_INC_DIR) -DUSE_MKL
#	LDFLAGS_ += -L$(MKL_LIB_DIR) -lmkl_sequential -lmkl_core -lmkl_intel_lp64
	LDFLAGS_ += -L$(MKL_LIB_DIR) -lmkl_sequential -lmkl_core -lmkl_rt
else ifeq ($(OPENBLAS_SUPPORT_),YES)
	CFLAGS_  += -I$(OPENBLAS_INC_DIR) -DUSE_OPENBLAS
	LDFLAGS_ += -L$(OPENBLAS_LIB_DIR) -lopenblas -Wl,-rpath=$(OPENBLAS_LIB_DIR)
endif

CFLAGS_ += -DVERBOSE -DFPGA_OTHER_LOOP_II=$(FPGA_OTHER_II) -DFPGA_GEMM_LOOP_II=$(FPGA_GEMM_II) -DBLOCK_SIZE=$(BLOCK_SIZE) -DFPGA_MEMORY_PORT_WIDTH=$(FPGA_MEMORY_PORT_WIDTH) -DSYRK_NUM_ACCS=$(SYRK_NUM_ACCS) -DGEMM_NUM_ACCS=$(GEMM_NUM_ACCS) -DTRSM_NUM_ACCS=$(TRSM_NUM_ACCS)

$(PROGRAM_)$(BLOCK_SIZE)-p: ./src/$(PROGRAM_).c
	$(CC) $(CFLAGS_) $^ -o $@ $(LDFLAGS_)

ait:
	ait -b alveo_u55c -c 300 -n cholesky -j 4 -v --disable_board_support_check --wrapper_version 13 --disable_spawn_queues --placement_file u55c_placement_$(GEMM_NUM_ACCS).json --floorplanning_constr all --slr_slices all --regslice_pipeline_stages 1:1:1 --interconnect_regslice all --enable_pom_axilite --max_deps_per_task=3 --max_args_per_task=3 --max_copies_per_task=3 --picos_tm_size=128 --picos_dm_size=390 --picos_vm_size=390 --from_step=$(FROM_STEP) --to_step=$(TOP_STEP)

info:
	@echo "========== OPENBLAS =========="
	@echo "  SUPPORT enabled:  $(OPENBLAS_SUPPORT_)"
	@echo "  OPENBLAS_DIR:     $(OPENBLAS_DIR)"
	@echo "  OPENBLAS_INC_DIR: $(OPENBLAS_INC_DIR)"
	@echo "  OPENBLAS_LIB_DIR: $(OPENBLAS_LIB_DIR)"
	@echo "  Headers:          $(if $(wildcard $(OPENBLAS_INC_DIR)/lapacke.h ),YES,NO)"
	@echo "  Lib files (.so):  $(if $(wildcard $(OPENBLAS_LIB_DIR)/libopenblas.so ),YES,NO)"
	@echo "=============================="
	@echo "============= MKL ============"
	@echo "  SUPPORT enabled:  $(MKL_SUPPORT_)"
	@echo "  MKL_DIR:          $(MKL_DIR)"
	@echo "  MKL_INC_DIR:      $(MKL_INC_DIR)"
	@echo "  MKL_LIB_DIR:      $(MKL_LIB_DIR)"
	@echo "  Headers:          $(if $(wildcard $(MKL_INC_DIR)/mkl.h ),YES,NO)"
	@echo "  Lib files (.so):  $(if $(wildcard $(MKL_LIB_DIR)/libmkl_sequential.so ),YES,NO)"
	@echo "=============================="
