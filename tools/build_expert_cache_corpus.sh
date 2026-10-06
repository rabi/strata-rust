# Build the expert_cache corpus: the real expert_cache.cpp, host-only, with the CUDA runtime and the
# driver's VMM entry points provided by the harness itself. See the harness header.
set -e
S=${STRATA_CPP:-/home/ramishra/work/LLM/Strata}
R=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$R/target/corpus"
g++ -std=c++20 -w -I"$S/include" -I"$R/tools/cudastub" -c "$S/src/core/expert_cache.cpp" -o "$R/target/corpus/ec.o"
g++ -std=c++20 -w -I"$S/include" -I"$R/tools/cudastub" -c "$R/tools/cudastub_defs.cpp" -o "$R/target/corpus/cstub.o"
g++ -std=c++20 -w -I"$S/include" -I"$R/tools/cudastub" -I"$R/tools" -c "$R/tools/expert_cache_corpus.cpp" -o "$R/target/corpus/ecc.o"
WRAPS=""
for f in cudaMalloc cudaFree cudaMemset cudaMemcpy cudaMemcpyAsync cudaDeviceSynchronize cudaGetLastError \
         cudaPeekAtLastError cudaGetErrorString cudaMemGetInfo cudaStreamSynchronize cudaGetDevice; do
    WRAPS="$WRAPS -Wl,--wrap=$f"
done
g++ -o "$R/target/corpus/eccorpus" "$R/target/corpus/ecc.o" "$R/target/corpus/ec.o" "$R/target/corpus/cstub.o" $WRAPS
