#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#define LPTB ((TSK*TSN)/(RTSM*RTSN))

// MAT OPERATORS
__kernel void filtercrop(const int coI, const int ciI, const int xI, const int yI,
						 const int I_displ, const int f_displ,
						 const int xc, const int yc,
						 __global floatX * Img,
						 __global floatX * OImg)
{
	const int I = get_global_id(0);
	const int J = get_global_id(1);
	//const int batch_id = get_global_id(2);
	floatX buff;
	
	for (int oc = 0; oc < coI; ++oc) {
		for (int ic = 0; ic < ciI; ++ic) {
			if (I >= xc && I < xI - xc && J >= yc && J < yI - yc) {
				buff = Img[(((oc * ciI + ic) * yI + J) * xI + I) * I_displ];
				OImg[(((oc * ciI + ic) * yI + J - yc) * xI + I - xc) * f_displ] = buff;
			}
		}
	}
}


__kernel void fconv2d(const int ciI, const int coI, const int xI, const int yI,
					  const int icf, const int ocf, const int xf, const int yf,
					  const int iI_displ, const int oI_displ,
					  const int padding,
					  __global floatX * oImg,
					  __global floatX * iImg,
					  __global floatX * df) 
{
	const int I = get_global_id(0); // 0 .. outX (= xI * 2 - 1)
	const int J = get_global_id(1); // 0 .. outY (= yI * 2 - 1)
	const int batch_id = 0; //get_global_id(2);
	floatX res;
	int xid;
	int yid;
	floatX oImg_buff;
	floatX iImg_buff;
	int fxs = xf / 2;
	int fys = yf / 2;
	int x_displ = xI * 2 - 1;
	int y_displ = yI * 2 - 1;
	
	// for each output channel
	for (int oc = 0; oc < ocf; ++oc) {
		// for each input channel
		for (int ic = 0; ic < icf; ++ic) {
			res = 0.0;
			for (int j = -fys; j < fys + (yf % 2); ++j) {
				for (int i = -fxs; i < fxs + (xf % 2); ++i) {
					xid = I + i - fxs;
					yid = J + j - fys;
					if (xid >= 0 && xid < xI && yid >= 0 && yid < yI) {
						oImg_buff = oImg[((oc * yI + yid) * xI + xid) * oI_displ + batch_id];
						iImg_buff = iImg[((ic * yI + j + fys) * xI + i + fxs) * iI_displ + batch_id];
					}
					else {
						if (padding == 0) {
							oImg_buff = 0.0;
							iImg_buff = 0.0;
						}
						else if (padding == 1) {
							xid = (xid % xI + xI) % xI;
							yid = (yid % yI + yI) % yI;
							oImg_buff = oImg[((oc * yI + yid) * xI + xid) * oI_displ + batch_id];
							iImg_buff = iImg[((ic * yI + j + fys) * xI + i + fxs) * iI_displ + batch_id];
						}
					}
					res += oImg_buff * iImg_buff;
				}
			}
			df[(((oc * icf + ic) * y_displ + y_displ - J - 1) * x_displ + x_displ - I - 1) * iI_displ] = res;
		}
	}
}


__kernel void conv2d(const int cI, const int xI, const int yI,
					 const int icf, const int ocf, const int xf, const int yf,
					 const int f_displ, const int img_displ,
					 const int padding,
					 __global floatX * Img,
					 __global floatX * f,
					 __global floatX * OImg) 
{
	const int I = get_global_id(0);
	const int J = get_global_id(1);
	const int batch_id = get_global_id(2);
	floatX res;
	floatX Ibuff;
	floatX fbuff;
	int xid;
	int yid;
	int fxs = xf / 2;
	int fys = yf / 2;
	// for each output channel
	for (int oc = 0; oc < ocf; ++oc) {
		res = 0.0;
		// for each input channel
		for (int ic = 0; ic < icf; ++ic) {
			// for each x, y if filter
			for (int j = -fys; j < fys + (yf % 2); ++j) {
				for (int i = -fxs; i < fxs + (xf % 2); ++i) {
					xid = I + i;
					yid = J + j;
					fbuff = f[(((oc * icf + ic) * yf + j + fys) * xf + i + fxs) * f_displ];
					if (xid >= 0 && xid < xI && yid >= 0 && yid < yI) {
						Ibuff = Img[((ic * yI + yid) * xI + xid) * img_displ + batch_id];
					}
					else {
						if (padding == 0) {
							Ibuff = 0.0;
						}
						else if (padding == 1) {
							xid = (xid % xI + xI) % xI;
							yid = (yid % yI + yI) % yI;
							//printf("xid: %d\tyid: %d", xid, yid);
							Ibuff = Img[((ic * yI + yid) * xI + xid) * img_displ + batch_id];
						}
					}
					//printf("%f\n", Img[(1 * xI + 1) * img_displ + 1]);
					//printf("I: %d\tJ: %d\ti: %d\tj: %d\tic: %d\toc: %d\tf= %f\tImg=%f\n", I, J, i, j, ic, oc, fbuff, Ibuff);
					res += Ibuff * fbuff;
				}
			}
		}
		//printf("I: %d\tJ: %d\t res: %f\n", I, J, res);
		OImg[((oc * yI + J) * xI + I) * img_displ + batch_id] = res;
	}
}


__kernel void matmul(const int M,
		const int N,
		const int K,
		__global floatX* A,
		__global floatX* B,
		__global floatX* C)
{
	// Thread identifiers
	const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
	const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
	const int offsetM = TSM * get_group_id(0); // Work-group offset
	const int offsetN = TSN * get_group_id(1); // Work-group offset

	// Local memory to fit a tile of A and B
	__local floatX Asub[TSK][TSM];
	__local floatX Bsub[TSN][TSK + 2];

	// Allocate register space
	floatX Areg;
	floatX Breg[WPTN];
	floatX acc[WPTM][WPTN];

	// Initialise the accumulation registers
	for (int wm = 0; wm < WPTM; ++wm) {
		for (int wn = 0; wn < WPTN; ++wn) {
			acc[wm][wn] = 0.0f;
		}
	}

	// Loop over all tiles
	int numTiles = K / TSK;
	for (int t = 0; t < numTiles; ++t) {

		// Load one tile of A and B into local memory
		for (int la = 0; la < LPTA; ++la) {
			int tid = tidn * RTSM + tidm;
			int id = la * RTSN * RTSM + tid;
			int row = id % TSM;
			int col = id / TSM;
			int tiledIndex = TSK * t + col;
			int Aid = tiledIndex * M + offsetM + row;
			int Bid = tiledIndex * N + offsetN + row;
			Asub[col][row] = A[Aid];
			Bsub[row][col] = B[Bid];
		}

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop over the values of a single tile
		for (int k = 0; k < TSK; ++k) {

			// Cache the values of Bsub in registers
			for (int wn = 0; wn < WPTN; ++wn) {
				int col = tidn + wn * RTSN;
				Breg[wn] = Bsub[col][k];
			}

			// Perform the computation
			for (int wm = 0; wm < WPTM; ++wm) {
				int row = tidm + wm * RTSM;
				Areg = Asub[k][row];
				for (int wn = 0; wn < WPTN; ++wn) {
					acc[wm][wn] += Areg * Breg[wn];
				}
			}
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the final results in C
	for (int wm = 0; wm < WPTM; ++wm) {
		int globalRow = offsetM + tidm + wm * RTSM;
		for (int wn = 0; wn < WPTN; ++wn) {
			int globalCol = offsetN + tidn + wn * RTSN;
			C[globalCol * M + globalRow] = acc[wm][wn];
		}
	}
}


__kernel void transpose(const int P, const int Q,
	__global floatX* input,
	__global floatX* output) {

	// Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..P
	const int ID1 = get_group_id(1) * TS + ty; // 0..Q

	// Set-up the local memory for shuffling
	__local floatX buffer[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < P && ID1 < Q) {
		buffer[ty][tx] = input[ID1 * P + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// We don't have to swap the x and y thread indices here,
	// because that's already done in the local memory
	const int newID0 = get_group_id(1) * TS + tx;
	const int newID1 = get_group_id(0) * TS + ty;

	// Store the transposed result (coalesced)
	if (newID0 < Q && newID1 < P) {
		output[newID1 * Q + newID0] = buffer[tx][ty];
	}
}


__kernel void matsum(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B,
	__global floatX* C)
{
	//    const int ID0 = get_global_id(0);
	//    const int ID1 = get_global_id(1);
	//    C[ID1 * M + ID0] = A[ID1 * M + ID0] + B[ID1 * M + ID0];
		 // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];
	__local floatX bufferB[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
		bufferB[tx][ty] = B[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		C[ID1 * M + ID0] = bufferA[tx][ty] + bufferB[tx][ty];
	}
}


__kernel void matsubstract(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B,
	__global floatX* C)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];
	__local floatX bufferB[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
		bufferB[tx][ty] = B[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		C[ID1 * M + ID0] = bufferA[tx][ty] - bufferB[tx][ty];
	}
}


__kernel void matdot(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B,
	__global floatX* C)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];
	__local floatX bufferB[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
		bufferB[tx][ty] = B[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		C[ID1 * M + ID0] = bufferA[tx][ty] * bufferB[tx][ty];
	}
}


__kernel void matscale(const int M,
	const int N,
	__global floatX* A,
	const floatX B,
	__global floatX* C)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		C[ID1 * M + ID0] = bufferA[tx][ty] * B;
	}
}


__kernel void matpluscol(const int M,
	const int N,
	const int K,
	__global floatX* A,
	__global floatX* b,
	__global floatX* C)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N
	const int IDb = get_global_id(0) * K;

	if (ID0 < M && ID1 < N) {
		C[ID0 * N + ID1] = A[ID0 * N + ID1] + b[IDb];
	}
}

// Activations and derivatives
__kernel void sigmoid(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		B[ID1 * M + ID0] = 1 / (1 + exp(-bufferA[tx][ty]));
	}
}

__kernel void dsigmoid(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		B[ID1 * M + ID0] = bufferA[tx][ty] * (1 - bufferA[tx][ty]);
	}
}


__kernel void relu(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		if (bufferA[tx][ty] > 0) {
			B[ID1 * M + ID0] = bufferA[tx][ty];
		}
		else {
			B[ID1 * M + ID0] = 0;
		}
	}
}


__kernel void drelu(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		if (bufferA[tx][ty] > 0) {
			B[ID1 * M + ID0] = 1.0;
		}
		else {
			B[ID1 * M + ID0] = 0.0;
		}
	}
}


__kernel void dsoftmax(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		B[ID1 * M + ID0] = bufferA[tx][ty] * (1 - bufferA[tx][ty]);
	}
}
