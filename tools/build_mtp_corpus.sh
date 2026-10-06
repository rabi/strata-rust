# Build the mtp corpus: the real mtp.cpp + the real layer.cpp, host-only, with the CUDA seam faked
# (--wrap) and the .cu-side size helpers transcribed in the harness.  See the harness header.
set -e
S=${STRATA_CPP:-/home/ramishra/work/LLM/Strata}
R=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$R/target/corpus"
g++ -std=c++20 -w -DGGML_COMMON_DECL_C -I"$S/include" -I"$S/third_party/ggml" -I"$R/tools/cudastub" \
    -c "$S/src/core/mtp.cpp" -o "$R/target/corpus/mtp.o"
g++ -std=c++20 -w -DGGML_COMMON_DECL_C -I"$S/include" -I"$S/third_party/ggml" -I"$R/tools/cudastub" \
    -c "$S/src/core/layer.cpp" -o "$R/target/corpus/layer.o"
g++ -std=c++20 -w -DGGML_COMMON_DECL_C -I"$S/include" -I"$S/third_party/ggml" -I"$R/tools/cudastub" \
    -c "$R/tools/mtp_corpus.cpp" -o "$R/target/corpus/mc.o"
python3 "$R/tools/mtp_stub_gen.py"
g++ -c "$R/tools/mtp_stub.s" -o "$R/target/corpus/mstub.o"
WRAPS=""
for f in cudaMalloc cudaFree cudaHostAlloc cudaFreeHost cudaMemcpy cudaMemcpyAsync cudaMemset \
         cudaMemsetAsync cudaDeviceSynchronize cudaGetLastError cudaGetDevice cudaSetDevice \
         cudaMemGetInfo cudaStreamCreateWithFlags cudaStreamDestroy cudaHostGetDevicePointer; do
    WRAPS="$WRAPS -Wl,--wrap=$f"
done
g++ -o "$R/target/corpus/mccorpus" "$R/target/corpus/mc.o" "$R/target/corpus/mtp.o" \
    "$R/target/corpus/layer.o" "$R/target/corpus/mstub.o" $WRAPS
