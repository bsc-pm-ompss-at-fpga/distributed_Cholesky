///////////////////
// Automatic IP Generated by OmpSs@FPGA compiler
///////////////////
// The below code is composed by:
//  1) User source code, which may be under any license (see in original source code)
//  2) OmpSs@FPGA toolchain code which is licensed under LGPLv3 terms and conditions
///////////////////
// Top IP Function: cholesky_blocked
// Accel. type hash: 6274650572
// Num. instances: 1
// Wrapper version: 13
///////////////////

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

static ap_uint<64> __mcxx_taskId;
template<class T>
union __mcxx_cast {
	unsigned long long int raw;
	T typed;
};
struct mcxx_inaxis {
	ap_uint<64> data;
};
typedef ap_axiu<64, 1, 1, 2> mcxx_outaxis;
struct __fpga_copyinfo_t {
	unsigned long long int copy_address;
	unsigned char arg_idx;
	unsigned char flags;
	unsigned int size;
};
struct __data_owner_info_t {
	unsigned long long int size;
	unsigned char owner;
	unsigned char depid;
};

void mcxx_task_create(const ap_uint<64> type, const ap_uint<8> instanceNum, const ap_uint<8> numArgs, const unsigned long long int args[], const ap_uint<8> numDeps, const unsigned long long int deps[], const ap_uint<8> numCopies, const __fpga_copyinfo_t copies[], int numDataOwners, __data_owner_info_t data_owners[], hls::stream<mcxx_outaxis>& mcxx_outPort, unsigned char ompif_rank, unsigned char ompif_size, unsigned char owner);
void mcxx_task_create(const ap_uint<64> type, const ap_uint<8> instanceNum, const ap_uint<8> numArgs, const unsigned long long int args[], const ap_uint<8> numDeps, const unsigned long long int deps[], const ap_uint<8> numCopies, const __fpga_copyinfo_t copies[], hls::stream<mcxx_outaxis>& mcxx_outPort);
void mcxx_taskwait(hls::stream<ap_uint<8> >& mcxx_spawnInPort, hls::stream<mcxx_outaxis>& mcxx_outPort);
template <typename T>
struct __mcxx_ptr_t {
	unsigned long long int val;
	__mcxx_ptr_t(unsigned long long int val) : val(val) {}
	__mcxx_ptr_t() {}
	inline operator __mcxx_ptr_t<const T>() const {
		return __mcxx_ptr_t<const T>(val);
	}
	template <typename V> inline __mcxx_ptr_t<T> operator+(V const val) const {
		return __mcxx_ptr_t<T>(this->val + val*sizeof(T));
	}
	template <typename V> inline __mcxx_ptr_t<T> operator-(V const val) const {
		return __mcxx_ptr_t<T>(this->val - val*sizeof(T));
	}
	template <typename V> inline operator V() const {
		return (V)val;
	}
};

unsigned char calc_owner(int i, int j, unsigned char size) {
#pragma HLS inline
	return (i + j)%size;
}

constexpr unsigned long long int ts = 256;

//We only store the lower matrix in column major order
static inline __mcxx_ptr_t<float> dep_addr(__mcxx_ptr_t<float> A, int i, int j, unsigned long long int nt, unsigned long long int tb) {
#pragma HLS inline
	const int irem = nt-i;
	return A + ((tb - (irem*(irem+1)/2)) + j - i)*ts*ts;
}

void cholesky_blocked_moved(const unsigned long long int nt, __mcxx_ptr_t<float> A, unsigned __ompif_rank, unsigned char __ompif_size, hls::stream<ap_uint<8> >& mcxx_spawnInPort, hls::stream<mcxx_outaxis>& mcxx_outPort)
{
#pragma HLS inline
	//Total number of blocks
	const unsigned long long int tb = (nt*(nt+1))/2;
	main_loop:
	for (int k = 0; k < nt; k++)
	{
		{
			unsigned long long int __mcxx_args[1L];
			unsigned long long int __mcxx_deps[1L];
			__fpga_copyinfo_t __mcxx_copies[1L];
			__mcxx_ptr_t<float> __mcxx_arg_0;
			__mcxx_arg_0 = dep_addr(A, k, k, nt, tb);
			__mcxx_args[0] = __mcxx_arg_0.val;
			const __fpga_copyinfo_t tmp_0 = {.copy_address = __mcxx_arg_0.val, .arg_idx = 0, .flags = 3, .size = ts*ts * sizeof(float)};
			__mcxx_copies[0] = tmp_0;
			__mcxx_ptr_t<float> __mcxx_dep_0;
			__mcxx_dep_0 = dep_addr(A, k, k, nt, tb);
			__mcxx_deps[0] = 3LLU << 58 | __mcxx_dep_0.val;
			mcxx_task_create(5840911080LLU, 255, 1, __mcxx_args, 1, __mcxx_deps, 1, __mcxx_copies, 0, 0, mcxx_outPort, __ompif_rank, __ompif_size, calc_owner(k, k, __ompif_size));
		}
		trsm_loop:
		for (int i = nt - 1; i >= k + 1; i--)
		{
			{
				unsigned long long int __mcxx_args[2L];
				unsigned long long int __mcxx_deps[2L];
				__fpga_copyinfo_t __mcxx_copies[2L];
				__mcxx_ptr_t<float> __mcxx_arg_0;
				__mcxx_arg_0 = dep_addr(A, k, k, nt, tb);
				__mcxx_args[0] = __mcxx_arg_0.val;
				const __fpga_copyinfo_t tmp_0 = {.copy_address = __mcxx_arg_0.val, .arg_idx = 0, .flags = 1, .size = ts*ts * sizeof(const float)};
				__mcxx_copies[0] = tmp_0;
				__mcxx_ptr_t<float> __mcxx_arg_1;
				__mcxx_arg_1 = dep_addr(A, k, i, nt, tb);
				__mcxx_args[1] = __mcxx_arg_1.val;
				const __fpga_copyinfo_t tmp_1 = {.copy_address = __mcxx_arg_1.val, .arg_idx = 1, .flags = 3, .size = ts*ts * sizeof(float)};
				__mcxx_copies[1] = tmp_1;
				__mcxx_ptr_t<float> __mcxx_dep_0;
				__mcxx_dep_0 = dep_addr(A, k, k, nt, tb);
				__mcxx_deps[0] = 1LLU << 58 | __mcxx_dep_0.val;
				__mcxx_ptr_t<float> __mcxx_dep_1;
				__mcxx_dep_1 = dep_addr(A, k, i, nt, tb);
				__mcxx_deps[1] = 3LLU << 58 | __mcxx_dep_1.val;
				__data_owner_info_t data_owners[1];
				const ap_uint<1> n_data_owners = nt-1-i < __ompif_size ? 1 : 0;
				const __data_owner_info_t owner_0 = {.size=ts*ts*sizeof(float), .owner=calc_owner(k, k,__ompif_size), .depid=0};
				data_owners[0] = owner_0;
				mcxx_task_create(6757884748LLU, 255, 2, __mcxx_args, 2, __mcxx_deps, 2, __mcxx_copies, n_data_owners, data_owners, mcxx_outPort, __ompif_rank, __ompif_size, calc_owner(k, i, __ompif_size));
			}
		}
		syrk_loop:
		for (int i = k + 1; i < nt; i++)
		{
			gemm_loop:
			for (int j = k + 1; j < i; j++)
			{
				{
					unsigned long long int __mcxx_args[3L];
					unsigned long long int __mcxx_deps[3L];
					__fpga_copyinfo_t __mcxx_copies[3L];
					__mcxx_ptr_t<float> __mcxx_arg_0;
					__mcxx_arg_0 = dep_addr(A, k, i, nt, tb);
					__mcxx_args[0] = __mcxx_arg_0.val;
					const __fpga_copyinfo_t tmp_0 = {.copy_address = __mcxx_arg_0.val, .arg_idx = 0, .flags = 1, .size = ts*ts * sizeof(const float)};
					__mcxx_copies[0] = tmp_0;
					__mcxx_ptr_t<float> __mcxx_arg_1;
					__mcxx_arg_1 =  dep_addr(A, k, j, nt, tb);
					__mcxx_args[1] = __mcxx_arg_1.val;
					const __fpga_copyinfo_t tmp_1 = {.copy_address = __mcxx_arg_1.val, .arg_idx = 1, .flags = 1, .size = ts*ts * sizeof(const float)};
					__mcxx_copies[1] = tmp_1;
					__mcxx_ptr_t<float> __mcxx_arg_2;
					__mcxx_arg_2 = dep_addr(A, j, i, nt, tb);
					__mcxx_args[2] = __mcxx_arg_2.val;
					const __fpga_copyinfo_t tmp_2 = {.copy_address = __mcxx_arg_2.val, .arg_idx = 2, .flags = 3, .size = ts*ts * sizeof(float)};
					__mcxx_copies[2] = tmp_2;
					__mcxx_ptr_t<float> __mcxx_dep_0;
					__mcxx_dep_0 = dep_addr(A, k, i, nt, tb);
					__mcxx_deps[0] = 1LLU << 58 | __mcxx_dep_0.val;
					__mcxx_ptr_t<float> __mcxx_dep_1;
					__mcxx_dep_1 = dep_addr(A, k, j, nt, tb);
					__mcxx_deps[1] = 1LLU << 58 | __mcxx_dep_1.val;
					__mcxx_ptr_t<float> __mcxx_dep_2;
					__mcxx_dep_2 = dep_addr(A, j, i, nt, tb);
					__mcxx_deps[2] = 3LLU << 58 | __mcxx_dep_2.val;
					__data_owner_info_t data_owners[2];
					const __data_owner_info_t owner_0 = {.size=ts*ts*sizeof(float), .owner=calc_owner(k, i, __ompif_size), .depid=0};
					const __data_owner_info_t owner_1 = {.size=ts*ts*sizeof(float), .owner=calc_owner(k, j, __ompif_size), .depid=1};
					ap_uint<2> n_data_owners = 0;
					if ((j-(k+1)) < __ompif_size) {
						data_owners[0] = owner_0;
						++n_data_owners;
					}
					int low = (k+1)%__ompif_size;
					int high = j%__ompif_size;
					int owner = i%__ompif_size;
					int cond = 0;
					if (high >= low) {
						cond = owner >= low && owner < high;
					}
					else {
						cond = owner >= low || owner < high;
					}
					if (i-(k+1) <= __ompif_size && calc_owner(j, i, __ompif_size) != calc_owner(j, j, __ompif_size) && !cond) {
						data_owners[n_data_owners] = owner_1;
						++n_data_owners;
					}
					mcxx_task_create(6757388164LLU, 255, 3, __mcxx_args, 3, __mcxx_deps, 3, __mcxx_copies, n_data_owners, data_owners, mcxx_outPort, __ompif_rank, __ompif_size, calc_owner(j, i, __ompif_size));
				}
			}
			{
				unsigned long long int __mcxx_args[2L];
				unsigned long long int __mcxx_deps[2L];
				__fpga_copyinfo_t __mcxx_copies[2L];
				__mcxx_ptr_t<float> __mcxx_arg_0;
				__mcxx_arg_0 = dep_addr(A, k, i, nt, tb);
				__mcxx_args[0] = __mcxx_arg_0.val;
				const __fpga_copyinfo_t tmp_0 = {.copy_address = __mcxx_arg_0.val, .arg_idx = 0, .flags = 1, .size = ts*ts * sizeof(const float)};
				__mcxx_copies[0] = tmp_0;
				__mcxx_ptr_t<float> __mcxx_arg_1;
				__mcxx_arg_1 = dep_addr(A, i, i, nt, tb);
				__mcxx_args[1] = __mcxx_arg_1.val;
				const __fpga_copyinfo_t tmp_1 = {.copy_address = __mcxx_arg_1.val, .arg_idx = 1, .flags = 3, .size = ts*ts * sizeof(float)};
				__mcxx_copies[1] = tmp_1;
				__mcxx_ptr_t<float> __mcxx_dep_0;
				__mcxx_dep_0 = dep_addr(A, k, i, nt, tb);
				__mcxx_deps[0] = 1LLU << 58 | __mcxx_dep_0.val;
				__mcxx_ptr_t<float> __mcxx_dep_1;
				__mcxx_dep_1 = dep_addr(A, i, i, nt, tb);
				__mcxx_deps[1] = 3LLU << 58 | __mcxx_dep_1.val;
				__data_owner_info_t data_owners[1];
				const __data_owner_info_t owner_0 = {.size=ts*ts*sizeof(float), .owner=calc_owner(k, i, __ompif_size), .depid=0};
				data_owners[0] = owner_0;
				const ap_uint<1> n_data_owners = i-(k+1) < __ompif_size ? 1 : 0;
				mcxx_task_create(6757855513LLU, 255, 2, __mcxx_args, 2, __mcxx_deps, 2, __mcxx_copies, n_data_owners, data_owners, mcxx_outPort, __ompif_rank, __ompif_size, calc_owner(i, i, __ompif_size));
			}
		}
	}
	mcxx_taskwait(mcxx_spawnInPort, mcxx_outPort);
}

void mcxx_write_out_port(const ap_uint<64> data, const ap_uint<2> dest, const ap_uint<1> last, hls::stream<mcxx_outaxis>& mcxx_outPort) {
#pragma HLS inline
	mcxx_outaxis axis_word;
	axis_word.data = data;
	axis_word.dest = dest;
	axis_word.last = last;
	mcxx_outPort.write(axis_word);
}

void cholesky_blocked_wrapper(hls::stream<ap_uint<64> >& mcxx_inPort, hls::stream<mcxx_outaxis>& mcxx_outPort, hls::stream<ap_uint<8> >& mcxx_spawnInPort, ap_uint<8> ompif_rank, ap_uint<8> ompif_size) {
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS interface axis port=mcxx_inPort
#pragma HLS interface axis port=mcxx_outPort
#pragma HLS interface axis port=mcxx_spawnInPort
#pragma HLS stable variable=ompif_rank
#pragma HLS stable variable=ompif_size
	mcxx_inPort.read(); //command word
	__mcxx_taskId = mcxx_inPort.read();
	ap_uint<64> __mcxx_parent_taskId = mcxx_inPort.read();
	unsigned long long int nt;
	__mcxx_ptr_t<float> A;
	{
#pragma HLS protocol fixed
		{
			ap_uint<8> mcxx_flags_0;
			ap_uint<64> mcxx_offset_0;
			mcxx_flags_0 = mcxx_inPort.read()(7,0);
			ap_wait();
			__mcxx_cast<unsigned long long int> mcxx_arg_0;
			mcxx_arg_0.raw = mcxx_inPort.read();
			nt = mcxx_arg_0.typed;
		}
		ap_wait();
		{
			ap_uint<8> mcxx_flags_1;
			ap_uint<64> mcxx_offset_1;
			mcxx_flags_1 = mcxx_inPort.read()(7,0);
			ap_wait();
			mcxx_offset_1 = mcxx_inPort.read();
			A.val = mcxx_offset_1;
		}
		ap_wait();
	}
	cholesky_blocked_moved(nt, A, ompif_rank, ompif_size, mcxx_spawnInPort, mcxx_outPort);
	{
#pragma HLS protocol fixed
		ap_uint<64> header = 0x03;
		ap_wait();
		mcxx_write_out_port(header, 0, 0, mcxx_outPort);
		ap_wait();
		mcxx_write_out_port(__mcxx_taskId, 0, 0, mcxx_outPort);
		ap_wait();
		mcxx_write_out_port(__mcxx_parent_taskId, 0, 1, mcxx_outPort);
		ap_wait();
	}
}

void mcxx_task_create(const ap_uint<64> type, const ap_uint<8> instanceNum, const ap_uint<8> numArgs, const unsigned long long int args[], const ap_uint<8> numDeps, const unsigned long long int deps[], const ap_uint<8> numCopies, const __fpga_copyinfo_t copies[], hls::stream<mcxx_outaxis>& mcxx_outPort) {
#pragma HLS inline
	const ap_uint<2> destId = 2;
	ap_uint<64> tmp;
	tmp(15,8)  = numArgs;
	tmp(23,16) = numDeps;
	tmp(31,24) = numCopies;
	mcxx_write_out_port(tmp, destId, 0, mcxx_outPort);
	mcxx_write_out_port(__mcxx_taskId, destId, 0, mcxx_outPort);
	tmp(47,40) = instanceNum;
	tmp(33,0)  = type(33,0);
	mcxx_write_out_port(tmp, destId, 0, mcxx_outPort);
	for (ap_uint<4> i = 0; i < numDeps(3,0); ++i) {
#pragma HLS unroll
		mcxx_write_out_port(deps[i], destId, numArgs == 0 && numCopies == 0 && i == numDeps-1, mcxx_outPort);
	}
	for (ap_uint<4> i = 0; i < numCopies(3,0); ++i) {
#pragma HLS unroll
		mcxx_write_out_port(copies[i].copy_address, destId, 0, mcxx_outPort);
		tmp(7,0) = copies[i].flags;
		tmp(15,8) = copies[i].arg_idx;
		tmp(63,32) = copies[i].size;
		mcxx_write_out_port(tmp, destId, numArgs == 0 && i == numCopies-1, mcxx_outPort);
	}
	for (ap_uint<4> i = 0; i < numArgs(3,0); ++i) {
#pragma HLS unroll
		mcxx_write_out_port(args[i], destId, i == numArgs-1, mcxx_outPort);
	}
}

void OMPIF_Send(const void *data, unsigned int size, int destination, const ap_uint<8> numDeps, const unsigned long long int deps[], hls::stream<mcxx_outaxis>& mcxx_outPort) {
#pragma HLS inline
	ap_uint<64> command;
	command(7,0) = 0; //SEND
	command(15,8) = 0; //tag
	command(23,16) = destination;
	command(63, 24) = (unsigned long long int)data;
	unsigned long long int args[2] = {command, (unsigned long long int)size};
	mcxx_task_create(4294967299LU, 0xFF, 2, args, numDeps, deps, 0, 0, mcxx_outPort);
}
void OMPIF_Recv(void *data, unsigned int size, int source, const ap_uint<8> numDeps, const unsigned long long int deps[], hls::stream<mcxx_outaxis>& mcxx_outPort) {
#pragma HLS inline
	ap_uint<64> command;
	command(7,0) = 0; //RECV
	command(15,8) = 0;
	command(23,16) = source;
	command(63, 24) = (unsigned long long int)data;
	unsigned long long int args[2] = {command, (unsigned long long int)size};
	mcxx_task_create(4294967300LU, 0xFF, 2, args, numDeps, deps, 0, 0, mcxx_outPort);
}

void mcxx_task_create(const ap_uint<64> type, const ap_uint<8> instanceNum, const ap_uint<8> numArgs, const unsigned long long int args[], const ap_uint<8> numDeps, const unsigned long long int deps[], const ap_uint<8> numCopies, const __fpga_copyinfo_t copies[], int numDataOwners, __data_owner_info_t data_owners[], hls::stream<mcxx_outaxis>& mcxx_outPort, unsigned char ompif_rank, unsigned char ompif_size, unsigned char owner) {
#pragma HLS inline
	const ap_uint<2> destId = 2;
	const unsigned char rank = ompif_rank;

	auto_sendrecv:
	for (ap_uint<4> i = 0; i < numDataOwners; ++i) {
		const ap_uint<64> depword = deps[data_owners[i].depid];
#pragma HLS unroll
		const bool is_in = depword[58];
		const unsigned char data_owner = data_owners[i].owner;
		const unsigned int size = data_owners[i].size;
		if (owner != rank && data_owners[i].owner == rank && is_in) {
			const unsigned long long addr = depword(55,0);
			const unsigned long long int dep[2] = {depword(55,0) | (1LLU << 58), 0x0000100000000000LLU | (3LLU << 58)};
			OMPIF_Send((void*)addr, size, owner, 2, dep, mcxx_outPort);
		}
		else if (owner == rank && data_owner != rank && is_in) {
			const unsigned long long addr = depword(55,0);
			const unsigned long long int dep[2] = {depword(55,0) | (2LLU << 58), 0x0000200000000000LLU | (3LLU << 58)};
			OMPIF_Recv((void*)addr, size, data_owner, 2, dep, mcxx_outPort);
		}
	}

	if (owner == rank) {
		mcxx_task_create(type, instanceNum, numArgs, args, numDeps, deps, numCopies, copies, mcxx_outPort);
	}

	/*for (ap_uint<4> i = 0; i < numDataOwners; ++i) {
#pragma HLS unroll
		const unsigned char data_owner = data_owners[i].owner;
		const unsigned int size = data_owners[i].size;
		const bool is_out = (deps[i] >> 59) & 0x1;
		if (owner == rank && data_owner == 255 && is_out) { //broadcast
			const unsigned long long addr = deps[i] & 0x00FFFFFFFFFFFFFF;
			const unsigned long long int dep[2] = {addr | (1LLU << 58), 0x0000100000000000LLU | (3LLU << 58)};
			unsigned char j = rank + 1 == ompif_size ? 0 : rank+1;
			OMPIF_Bcast((void*)addr, size, 2, dep, mcxx_outPort);
		}
		else if (owner != rank && data_owner == 255 && is_out) {
			const unsigned long long addr = deps[i] & 0x00FFFFFFFFFFFFFF;
			const unsigned long long int dep[2] = {addr | (2LLU << 58), 0x0000200000000000LLU | (3LLU << 58)};
			OMPIF_Recv((void*)addr, size, owner, 2, dep, mcxx_outPort);
		}
	}*/
}

void mcxx_taskwait(hls::stream<ap_uint<8> >& mcxx_spawnInPort, hls::stream<mcxx_outaxis>& mcxx_outPort) {
#pragma HLS inline
	ap_wait();
	mcxx_write_out_port(__mcxx_taskId, 3, 1, mcxx_outPort);
	ap_wait();
	mcxx_spawnInPort.read();
	ap_wait();
}
