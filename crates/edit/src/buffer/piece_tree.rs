// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

use std::ops::Range;
use std::ptr::NonNull;
use std::{io, slice};

use stdext::ReplaceRange as _;
use stdext::arena::Arena;

use crate::document::{ReadableDocument, WriteableDocument};
use crate::helpers::*;

#[cfg(target_pointer_width = "32")]
const LARGE_CAPACITY: usize = 512 * MEBI;
#[cfg(target_pointer_width = "64")]
const LARGE_CAPACITY: usize = 4 * GIBI;
const SMALL_CAPACITY: usize = 128 * MEBI;

type NodePtr = Option<NonNull<Node>>;
type RevisionPtr<T> = NonNull<Revision<T>>;

#[derive(Clone, Copy)]
struct Piece {
    text: NonNull<u8>,
    len: usize,
}

#[derive(Clone, Copy)]
enum NodeKind {
    Leaf(Piece),
    Branch { left: NonNull<Node>, right: NonNull<Node> },
}

struct Node {
    kind: NodeKind,
    len: usize,
    height: u8,
}

struct Revision<T> {
    root: NodePtr,
    metadata: T,
    generation_before: u32,
    previous: Option<RevisionPtr<T>>,
    redo_next: Option<RevisionPtr<T>>,
}

/// An arena-backed piece tree with persistent revisions.
///
/// Tree nodes and inserted text are never freed individually. Starting an edit creates a
/// revision whose root shares all unchanged nodes with the previous revision. `T` is scratch
/// space owned by that revision, intended for editor state needed by undo and redo.
pub struct PieceTree<T: Copy> {
    arena: Arena,
    current: RevisionPtr<T>,
    redo: Option<RevisionPtr<T>>,
    generation: u32,
}

impl<T: Copy> PieceTree<T> {
    pub fn new(small: bool, metadata: T) -> io::Result<Self> {
        let arena = Arena::new(if small { SMALL_CAPACITY } else { LARGE_CAPACITY })?;
        let revision = arena.alloc_uninit().write(Revision {
            root: None,
            metadata,
            generation_before: 0,
            previous: None,
            redo_next: None,
        });

        Ok(Self { current: NonNull::from(revision), arena, redo: None, generation: 0 })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.root().map_or(0, |root| unsafe { root.as_ref().len })
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Starts an edit revision. When `coalesce` is true, the current revision remains the
    /// pending head and subsequent replacements become part of the same undo step.
    pub fn begin_revision(&mut self, metadata: T, coalesce: bool) {
        self.redo = None;
        if coalesce && unsafe { self.current.as_ref().previous.is_some() } {
            return;
        }

        let revision = self.arena.alloc_uninit().write(Revision {
            root: self.root(),
            metadata,
            generation_before: self.generation,
            previous: Some(self.current),
            redo_next: None,
        });
        self.current = NonNull::from(revision);
    }

    pub fn revision_metadata_mut(&mut self) -> &mut T {
        unsafe { &mut self.current.as_mut().metadata }
    }

    /// Discards all reachable history while retaining the current root and arena allocations.
    pub fn reset_history(&mut self, metadata: T) {
        let revision = self.arena.alloc_uninit().write(Revision {
            root: self.root(),
            metadata,
            generation_before: self.generation,
            previous: None,
            redo_next: None,
        });
        self.current = NonNull::from(revision);
        self.redo = None;
    }

    pub fn undo(&mut self, current_metadata: T) -> Option<T> {
        let mut revision = self.current;
        let previous = unsafe { revision.as_ref().previous }?;
        let revision = unsafe { revision.as_mut() };

        let metadata = revision.metadata;
        revision.metadata = current_metadata;
        std::mem::swap(&mut self.generation, &mut revision.generation_before);
        revision.redo_next = self.redo;
        self.redo = Some(self.current);
        self.current = previous;
        Some(metadata)
    }

    pub fn redo(&mut self, current_metadata: T) -> Option<T> {
        let mut revision_ptr = self.redo?;
        let revision = unsafe { revision_ptr.as_mut() };

        let metadata = revision.metadata;
        revision.metadata = current_metadata;
        std::mem::swap(&mut self.generation, &mut revision.generation_before);
        self.redo = revision.redo_next.take();
        self.current = revision_ptr;
        Some(metadata)
    }

    pub fn extract_raw(&self, range: Range<usize>, out: &mut Vec<u8>, mut out_off: usize) {
        let end = range.end.min(self.len());
        let mut beg = range.start.min(end);
        out_off = out_off.min(out.len());
        out.reserve(end - beg);

        while beg < end {
            let chunk = self.read_forward(beg);
            let chunk = &chunk[..chunk.len().min(end - beg)];
            out.replace_range(out_off..out_off, chunk);
            beg += chunk.len();
            out_off += chunk.len();
        }
    }

    pub fn clear(&mut self) {
        self.set_root(None);
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn copy_from(&mut self, src: &dyn ReadableDocument) -> bool {
        let mut offset = 0;
        loop {
            let dst = self.read_forward(offset);
            let src = src.read_forward(offset);
            let len = dst.len().min(src.len());
            if dst[..len] != src[..len] {
                break;
            }
            if len == 0 {
                if dst.len() == src.len() {
                    return false;
                }
                break;
            }
            offset += len;
        }

        let mut replacement = Vec::new();
        let mut src_offset = offset;
        loop {
            let chunk = src.read_forward(src_offset);
            if chunk.is_empty() {
                break;
            }
            replacement.extend_from_slice(chunk);
            src_offset += chunk.len();
        }
        self.replace(offset..usize::MAX, &replacement);
        true
    }

    pub fn copy_into(&self, dst: &mut dyn WriteableDocument) {
        let mut source_offset = 0;
        let mut destination_offset = 0;
        while {
            let chunk = self.read_forward(source_offset);
            dst.replace(destination_offset..usize::MAX, chunk);
            destination_offset = usize::MAX;
            source_offset += chunk.len();
            source_offset < self.len()
        } {}
    }

    fn root(&self) -> NodePtr {
        unsafe { self.current.as_ref().root }
    }

    fn set_root(&mut self, root: NodePtr) {
        unsafe { self.current.as_mut().root = root };
    }

    fn alloc_leaf(&self, piece: Piece) -> NonNull<Node> {
        NonNull::from(self.arena.alloc_uninit().write(Node {
            kind: NodeKind::Leaf(piece),
            len: piece.len,
            height: 1,
        }))
    }

    fn alloc_branch(&self, left: NonNull<Node>, right: NonNull<Node>) -> NonNull<Node> {
        let left_ref = unsafe { left.as_ref() };
        let right_ref = unsafe { right.as_ref() };
        NonNull::from(self.arena.alloc_uninit().write(Node {
            kind: NodeKind::Branch { left, right },
            len: left_ref.len + right_ref.len,
            height: left_ref.height.max(right_ref.height) + 1,
        }))
    }

    fn join(&self, left: NodePtr, right: NodePtr) -> NodePtr {
        let (Some(left), Some(right)) = (left, right) else {
            return left.or(right);
        };
        let left_ref = unsafe { left.as_ref() };
        let right_ref = unsafe { right.as_ref() };

        if left_ref.height > right_ref.height + 1 {
            let NodeKind::Branch { left: ll, right: lr } = left_ref.kind else {
                unreachable!();
            };
            return self.balance(ll, self.join(Some(lr), Some(right)).unwrap());
        }
        if right_ref.height > left_ref.height + 1 {
            let NodeKind::Branch { left: rl, right: rr } = right_ref.kind else {
                unreachable!();
            };
            return self.balance(self.join(Some(left), Some(rl)).unwrap(), rr);
        }

        Some(self.alloc_branch(left, right))
    }

    fn balance(&self, left: NonNull<Node>, right: NonNull<Node>) -> NodePtr {
        let left_height = unsafe { left.as_ref().height };
        let right_height = unsafe { right.as_ref().height };

        if left_height > right_height + 1 {
            let NodeKind::Branch { left: ll, right: lr } = (unsafe { left.as_ref().kind }) else {
                unreachable!();
            };
            if unsafe { ll.as_ref().height >= lr.as_ref().height } {
                return Some(self.alloc_branch(ll, self.alloc_branch(lr, right)));
            }
            let NodeKind::Branch { left: lrl, right: lrr } = (unsafe { lr.as_ref().kind }) else {
                unreachable!();
            };
            return Some(
                self.alloc_branch(self.alloc_branch(ll, lrl), self.alloc_branch(lrr, right)),
            );
        }
        if right_height > left_height + 1 {
            let NodeKind::Branch { left: rl, right: rr } = (unsafe { right.as_ref().kind }) else {
                unreachable!();
            };
            if unsafe { rr.as_ref().height >= rl.as_ref().height } {
                return Some(self.alloc_branch(self.alloc_branch(left, rl), rr));
            }
            let NodeKind::Branch { left: rll, right: rlr } = (unsafe { rl.as_ref().kind }) else {
                unreachable!();
            };
            return Some(
                self.alloc_branch(self.alloc_branch(left, rll), self.alloc_branch(rlr, rr)),
            );
        }

        Some(self.alloc_branch(left, right))
    }

    fn split(&self, root: NodePtr, offset: usize) -> (NodePtr, NodePtr) {
        let Some(root) = root else {
            return (None, None);
        };
        let node = unsafe { root.as_ref() };
        let offset = offset.min(node.len);

        match node.kind {
            NodeKind::Leaf(piece) => {
                if offset == 0 {
                    (None, Some(root))
                } else if offset == piece.len {
                    (Some(root), None)
                } else {
                    let left = Piece { text: piece.text, len: offset };
                    let right =
                        Piece { text: unsafe { piece.text.add(offset) }, len: piece.len - offset };
                    (Some(self.alloc_leaf(left)), Some(self.alloc_leaf(right)))
                }
            }
            NodeKind::Branch { left, right } => {
                let left_len = unsafe { left.as_ref().len };
                if offset < left_len {
                    let (beg, end) = self.split(Some(left), offset);
                    (beg, self.join(end, Some(right)))
                } else {
                    let (beg, end) = self.split(Some(right), offset - left_len);
                    (self.join(Some(left), beg), end)
                }
            }
        }
    }

    fn piece_at(&self, mut offset: usize) -> Option<(Piece, usize)> {
        let mut node = self.root()?;
        loop {
            match unsafe { node.as_ref().kind } {
                NodeKind::Leaf(piece) => return Some((piece, offset)),
                NodeKind::Branch { left, right } => {
                    let left_len = unsafe { left.as_ref().len };
                    if offset < left_len {
                        node = left;
                    } else {
                        offset -= left_len;
                        node = right;
                    }
                }
            }
        }
    }
}

impl<T: Copy> ReadableDocument for PieceTree<T> {
    fn read_forward(&self, off: usize) -> &[u8] {
        let off = off.min(self.len());
        if off == self.len() {
            return &[];
        }
        let (piece, offset) = self.piece_at(off).unwrap();
        unsafe { slice::from_raw_parts(piece.text.add(offset).as_ptr(), piece.len - offset) }
    }

    fn read_backward(&self, off: usize) -> &[u8] {
        let off = off.min(self.len());
        if off == 0 {
            return &[];
        }
        let (piece, offset) = self.piece_at(off - 1).unwrap();
        unsafe { slice::from_raw_parts(piece.text.as_ptr(), offset + 1) }
    }
}

impl<T: Copy> WriteableDocument for PieceTree<T> {
    fn replace(&mut self, range: Range<usize>, replacement: &[u8]) {
        let len = self.len();
        let beg = range.start.min(len);
        let end = range.end.min(len).max(beg);
        let root = self.root();
        let (left, rest) = self.split(root, beg);
        let (_, right) = self.split(rest, end - beg);

        let inserted = if replacement.is_empty() {
            None
        } else {
            let text = self.arena.alloc_slice(replacement.len(), 0u8);
            text.copy_from_slice(replacement);
            Some(self.alloc_leaf(Piece {
                text: NonNull::new(text.as_mut_ptr()).unwrap(),
                len: text.len(),
            }))
        };

        let root = self.join(self.join(left, inserted), right);
        self.set_root(root);
        self.generation = self.generation.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::PieceTree;
    use crate::document::{ReadableDocument as _, WriteableDocument as _};

    fn contents<T: Copy>(tree: &PieceTree<T>) -> Vec<u8> {
        let mut result = Vec::new();
        tree.extract_raw(0..usize::MAX, &mut result, 0);
        result
    }

    #[test]
    fn replace_and_read_across_pieces() {
        let mut tree = PieceTree::new(true, ()).unwrap();
        tree.replace(0..0, b"abcd");
        tree.replace(2..2, b"XY");
        tree.replace(1..5, b"!");

        assert_eq!(contents(&tree), b"a!d");
        assert_eq!(tree.read_forward(1), b"!");
        assert_eq!(tree.read_backward(3), b"d");
    }

    #[test]
    fn revisions_restore_roots_and_metadata() {
        let mut tree = PieceTree::new(true, 0).unwrap();
        tree.begin_revision(10, false);
        tree.replace(0..0, b"a");
        tree.begin_revision(20, true);
        tree.replace(1..1, b"b");
        tree.begin_revision(20, true);
        tree.replace(2..2, b"c");

        assert_eq!(contents(&tree), b"abc");
        assert_eq!(tree.undo(30), Some(10));
        assert_eq!(contents(&tree), b"");
        assert_eq!(tree.redo(10), Some(30));
        assert_eq!(contents(&tree), b"abc");
    }

    #[test]
    fn randomized_replacements_match_vec() {
        let mut tree = PieceTree::new(true, ()).unwrap();
        let mut expected = Vec::new();
        let mut random = 0x1234_5678_u32;

        for _ in 0..2000 {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let beg = (random as usize) % (expected.len() + 1);
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let end = beg + (random as usize) % (expected.len() - beg + 1);
            let replacement = [b'a' + (random % 26) as u8, b'0' + (random % 10) as u8];
            let replacement = &replacement[..(random as usize >> 8) % 3];

            expected.splice(beg..end, replacement.iter().copied());
            tree.replace(beg..end, replacement);
            assert_eq!(contents(&tree), expected);
        }
    }
}
