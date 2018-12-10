# test various training parameters

bigdata = joinpath(dirname(@__FILE__),"..", "data", "big.txt")

TMPDIR = "./tmp"
mkpath(TMPDIR)

CORPUS = bigdata
VOCAB_FILE = joinpath(TMPDIR, "vocab.txt")
COOCCURRENCE_FILE = joinpath(TMPDIR, "cooccurrence.bin")
COOCCURRENCE_SHUF_FILE = joinpath(TMPDIR, "cooccurrence.shuf.bin")
SAVE_FILE = joinpath(TMPDIR, "vectors")
VERBOSE = 2
MEMORY = 4.0
VOCAB_MIN_COUNT = 5
VECTOR_SIZE = rand(10:50)
MAX_ITER = rand(3:5)
WINDOW_SIZE = rand(5:15)
BINARY_OPTS = [0,1]
NUM_THREADS = 1
X_MAX = 10.0
HEADER_OPTS = [0,1]

vocab_count(CORPUS, VOCAB_FILE, min_count=VOCAB_MIN_COUNT, verbose=VERBOSE)
cooccur(CORPUS, VOCAB_FILE, COOCCURRENCE_FILE,
        memory=MEMORY, verbose=VERBOSE, window_size=WINDOW_SIZE)
shuffle(COOCCURRENCE_FILE, COOCCURRENCE_SHUF_FILE,
        memory=MEMORY, verbose=VERBOSE)
for BINARY in BINARY_OPTS
    for HEADER in HEADER_OPTS
        glove(COOCCURRENCE_SHUF_FILE, VOCAB_FILE, SAVE_FILE,
              threads=NUM_THREADS, x_max=X_MAX, iter=MAX_ITER,
              vector_size=VECTOR_SIZE, binary=BINARY,
              write_header=HEADER, verbose=VERBOSE)
    end
end

#rm(TMPDIR, recursive=true, force=true)

println("training passed test...")

