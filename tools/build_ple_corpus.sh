# Build the ple_reader corpus: the real ple_reader.cpp, host-only, with the platform layer
# (DirectFile + the clock) provided by the harness itself. See the harness header.
set -e
S=${STRATA_CPP:-/home/ramishra/work/LLM/Strata}
R=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$R/target/corpus"
g++ -std=c++20 -w -I"$S/include" -I"$S/src" -c "$R/tools/ple_corpus.cpp" -o "$R/target/corpus/plec.o"
g++ -o "$R/target/corpus/plecorpus" "$R/target/corpus/plec.o"
