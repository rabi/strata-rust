//! The container the snapshot's RAM accounting is written against.
//!
//! `ConversationCheckpoint::bytes()`, `ConversationKvReuse::bytes()` and
//! `SavedConversation::bytes()` are the numbers admission is decided on, and they
//! count `capacity() * sizeof(T)` for every member vector. So the answer depends
//! on how the container GROWS, not only on what it holds:
//!
//! ```text
//! reuse.kv after reserve(3) + 3 push_backs   capacity 3  ->  3 * 216 =  648 B
//! reuse.kv with 3 unreserved push_backs       capacity 4  ->  4 * 216 =  864 B
//! ```
//!
//! which is the difference between the golden `reuse_bytes=10248` and 10464 for
//! one fixture. libstdc++ doubles from 1 (`push_back` on empty gives 1, 2, 4, 8 …)
//! and `reserve(n)` allocates exactly `n`; Rust's `Vec` starts a `push_back` at 4
//! and rounds capacities up, so a plain `Vec` would report different RAM for the
//! same snapshot. This type reproduces the libstdc++ growth policy that the C++
//! numbers were measured against; the payload itself is an ordinary `Vec`.
//!
//! Every rule below is checked against the C++ in `tests/conversation_corpus.rs`
//! (`image_bytes`, `reuse_bytes`, `bytes` fields of the golden lines).

/// `sizeof` of the C++ types whose vectors are counted in `bytes()`. Measured
/// from the real headers (`/tmp/conv_oracle/sizes2`), libstdc++/x86-64.
pub const SIZEOF_IMAGE_KEY: usize = 16;
pub const SIZEOF_CHECKPOINT: usize = 200;
pub const SIZEOF_KV: usize = 216;
pub const SIZEOF_KV_REUSE: usize = 64;
pub const SIZEOF_SAVED: usize = 440;

/// A `std::vector<T>` with libstdc++'s growth policy and C++ `bytes()` cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CppVec<T> {
    items: Vec<T>,
    /// The modelled libstdc++ capacity. `items.capacity()` is deliberately not
    /// used: it follows Rust's allocator policy, which is not this one.
    cap: usize,
    /// `sizeof(T)` as the C++ counts it (the C++ object, not the Rust replica).
    unit: usize,
}

impl<T: Clone> Default for CppVec<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            cap: 0,
            unit: std::mem::size_of::<T>(),
        }
    }
}

impl<T: Clone> CppVec<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// A vector built with an explicit `unit` size, for a Rust replica whose
    /// fields are laid out differently from the C++ struct it stands for.
    pub fn with_unit(unit: usize) -> Self {
        Self {
            items: Vec::new(),
            cap: 0,
            unit,
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The modelled capacity, i.e. what C++ `capacity()` would report.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.items.get(i)
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.items.get_mut(i)
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.items.last_mut()
    }

    pub fn push(&mut self, value: T) {
        if self.items.len() == self.cap {
            self.cap = (self.cap * 2).max(1);
        }
        self.items.push(value);
    }

    /// C++ `reserve(n)`: exactly `n`, never more.
    pub fn reserve(&mut self, n: usize) {
        if n > self.cap {
            self.cap = n;
        }
    }

    /// C++ `resize(n, value)`. Growing goes through `_M_check_len`, which is
    /// `size + max(size, need)` — the vector's SIZE, not its capacity — and
    /// shrinking keeps the capacity it already has.
    pub fn resize(&mut self, n: usize, value: &T) {
        let len = self.items.len();
        if n > len {
            let need = n - len;
            if n > self.cap {
                self.cap = len + len.max(need);
            }
        }
        self.items.resize(n, value.clone());
    }

    /// C++ `pop_back()`: shrinks the size and keeps the capacity it has.
    pub fn pop(&mut self) {
        self.items.pop();
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Bytes held: modelled capacity x `sizeof(T)`. The C++ counts the directory
    /// only, never the elements' own heap payloads — those are added by the
    /// element's own `bytes()`.
    pub fn bytes(&self) -> usize {
        self.cap * self.unit
    }
}
