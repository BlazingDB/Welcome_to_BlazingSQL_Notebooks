# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoders for text data.

* TextEncoder: base class
* SubwordTextEncoder: invertible
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from itertools import chain
import re
import time
import six
from six.moves import range  # pylint: disable=redefined-builtin

# Reserved tokens for things like padding and EOS symbols.
PAD = "[PAD]"
EOS = "[EOS]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
RESERVED_TOKENS = [PAD, EOS, UNK, CLS, SEP, MASK]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1

if six.PY2:
  RESERVED_TOKENS_BYTES = RESERVED_TOKENS
else:
  RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii")]

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")
_SPECIAL_CHARS = set(u"!\"\'#$%&*()`+,-./:;<=>?@[]^_{}~|")

# Unicode utility functions that work with Python 2 and 3
def native_to_unicode(s):
  if is_unicode(s):
    return s
  try:
    return to_unicode(s)
  except UnicodeDecodeError:
    res = to_unicode(s, ignore_errors=True)
#     tf.logging.info("Ignoring Unicode error, outputting: %s" % res)
    return res


def unicode_to_native(s):
  if six.PY2:
    return s.encode("utf-8") if is_unicode(s) else s
  else:
    return s


def is_unicode(s):
  return isinstance(s, six.text_type)


def to_unicode(s, ignore_errors=False):
  if is_unicode(s):
    return s
  error_mode = "ignore" if ignore_errors else "strict"
  return s.decode("utf-8", errors=error_mode)


class TextEncoder(object):
  """Base class for converting from ints to/from human readable strings."""

  def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
    self._num_reserved_ids = num_reserved_ids

  @property
  def num_reserved_ids(self):
    return self._num_reserved_ids


  @property
  def vocab_size(self):
    raise NotImplementedError()


def _escape_token(token, alphabet):
  """Escape away underscores and OOV characters and append '_'.

  This allows the token to be expressed as the concatenation of a list
  of subtokens from the vocabulary. The underscore acts as a sentinel
  which allows us to invertibly concatenate multiple such lists.

  Args:
    token: A unicode string to be escaped.
    alphabet: A set of all characters in the vocabulary's alphabet.

  Returns:
    escaped_token: An escaped unicode string.

  Raises:
    ValueError: If the provided token is not unicode.
  """
  if not isinstance(token, six.text_type):
    raise ValueError("Expected string type for token, got %s" % type(token))

  token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
  ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
  return u"".join(ret) + "_"

def _my_escape_token(token, alphabet):

  if not isinstance(token, six.text_type):
    raise ValueError("Expected string type for token, got %s" % type(token))

  token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
  ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
  return "_" + u"".join(ret)


class SubwordTextEncoder(TextEncoder):
  """Class for invertibly encoding text using a limited vocabulary.

  Invertibly encodes a native string as a sequence of subtokens from a limited
  vocabulary.

  A SubwordTextEncoder is built from a corpus (so it is tailored to the text in
  the corpus), and stored to a file. See text_encoder_build_subword.py.

  It can then be loaded and used to encode/decode any text.

  Encoding has four phases:

  1. Tokenize into a list of tokens.  Each token is a unicode string of either
     all alphanumeric characters or all non-alphanumeric characters.  We drop
     tokens consisting of a single space that are between two alphanumeric
     tokens.

  2. Escape each token.  This escapes away special and out-of-vocabulary
     characters, and makes sure that each token ends with an underscore, and
     has no other underscores.

  3. Represent each escaped token as a the concatenation of a list of subtokens
     from the limited vocabulary.  Subtoken selection is done greedily from
     beginning to end.  That is, we construct the list in order, always picking
     the longest subtoken in our vocabulary that matches a prefix of the
     remaining portion of the encoded token.

  4. Concatenate these lists.  This concatenation is invertible due to the
     fact that the trailing underscores indicate when one list is finished.

  """

  def __init__(self, filename=None):
    """Initialize and read from a file, if provided.

    Args:
      filename: filename from which to read vocab. If None, do not load a
        vocab
    """
    self._alphabet = set()
    super(SubwordTextEncoder, self).__init__()

  @property
  def vocab_size(self):
    """The subtoken vocabulary size."""
    return len(self._all_subtoken_strings)

  def _escaped_token_to_subtoken_strings(self, escaped_token):
    """Converts an escaped token string to a list of subtoken strings.

    Args:
      escaped_token: An escaped token as a unicode string.
    Returns:
      A list of subtokens as unicode strings.
    """
    # NOTE: This algorithm is greedy; it won't necessarily produce the "best"
    # list of subtokens.
    ret = []
    start = 0
    token_len = len(escaped_token)
    while start < token_len:
      for end in range(
          min(token_len, start + self._max_subtoken_len), start, -1):
        subtoken = escaped_token[start:end]
        if subtoken in self._subtoken_string_to_id:
          ret.append(subtoken)
          start = end
          break

      else:  # Did not break
        # If there is no possible encoding of the escaped token then one of the
        # characters in the token is not in the alphabet. This should be
        # impossible and would be indicative of a bug.
        assert False, "Token substring not found in subtoken vocabulary."

    return ret

  def build_from_token_counts(self,
                              token_counts,
                              min_count,
                              num_iterations=4,
                              reserved_tokens=None,
                              max_subtoken_length=None):
    """Train a SubwordTextEncoder based on a dictionary of word counts.

    Args:
      token_counts: a dictionary of Unicode strings to int.
      min_count: an integer - discard subtokens with lower counts.
      num_iterations: an integer.  how many iterations of refinement.
      reserved_tokens: List of reserved tokens. The global variable
        `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
        argument is `None`, it will use `RESERVED_TOKENS`.
      max_subtoken_length: Maximum length of a subtoken. If this is not set,
        then the runtime and memory use of creating the vocab is quadratic in
        the length of the longest token. If this is set, then it is instead
        O(max_subtoken_length * length of longest token).

    Raises:
      ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
        is not clear what the space is being reserved for, or when it will be
        filled in.
    """
    if reserved_tokens is None:
      reserved_tokens = RESERVED_TOKENS
    else:
      # There is not complete freedom in replacing RESERVED_TOKENS.
      for default, proposed in zip(RESERVED_TOKENS, reserved_tokens):
        if default != proposed:
          raise ValueError("RESERVED_TOKENS must be a prefix of "
                           "reserved_tokens.")

    start_time = time.time()

    # Initialize the alphabet. Note, this must include reserved tokens or it can
    # result in encoding failures.
    alphabet_tokens = chain(six.iterkeys(token_counts),
                            [native_to_unicode(t) for t in reserved_tokens])

    # all alphabets in tokens
    self._init_alphabet_from_tokens(alphabet_tokens)

    # Bootstrap the initial list of subtokens with the characters from the
    # alphabet plus the escaping characters.
    self._init_subtokens_from_list(list(self._alphabet),
                                   reserved_tokens=reserved_tokens)

    # We build iteratively.  On each iteration, we segment all the words,
    # then count the resulting potential subtokens, keeping the ones
    # with high enough counts for our new vocabulary.
    if min_count < 1:
      min_count = 1
    for i in range(num_iterations):

      # Collect all substrings of the encoded token that break along current
      # subtoken boundaries.
      subtoken_counts = collections.defaultdict(int)
    
      for token, count in six.iteritems(token_counts):
        iter_start_time = time.time()
        escaped_token = _my_escape_token(token, self._alphabet)
        subtokens = self._escaped_token_to_subtoken_strings(escaped_token)

        # excaped_token '_1234' -> subtoknes ['_12', '34'] (ex)
        # '_1234':100 -> '_', '_1', '_12', '_123', '_1234','3', '34' :+= 100,
        start = 0
        for subtoken in subtokens:
          last_position = len(escaped_token) + 1
          if max_subtoken_length is not None:
            last_position = min(last_position, start + max_subtoken_length)

          for end in range(start + 1, last_position):
            new_subtoken = escaped_token[start:end]
            subtoken_counts[new_subtoken] += count
          start += len(subtoken)

        iter_time_secs = time.time() - iter_start_time
 
      # Array of sets of candidate subtoken strings, by length.
      len_to_subtoken_strings = []
      for subtoken_string, count in six.iteritems(subtoken_counts):
        lsub = len(subtoken_string)
        if count >= min_count:
          while len(len_to_subtoken_strings) <= lsub:
            len_to_subtoken_strings.append(set())
          len_to_subtoken_strings[lsub].add(subtoken_string)

      # Consider the candidates longest to shortest, so that if we accept
      # a longer subtoken string, we can decrement the counts of its prefixes.
      new_subtoken_strings_with_count = []
      for lsub in range(len(len_to_subtoken_strings) - 1, 0, -1):
        subtoken_strings = len_to_subtoken_strings[lsub]
        for subtoken_string in subtoken_strings:
          count = subtoken_counts[subtoken_string]
          if count >= min_count:
            # Exclude alphabet tokens here, as they must be included later,
            # explicitly, regardless of count.
            if subtoken_string not in self._alphabet:
              new_subtoken_strings_with_count.append((count, subtoken_string))
            for l in range(1, lsub):
              subtoken_counts[subtoken_string[:l]] -= count

      # Include the alphabet explicitly to guarantee all strings are encodable.
      new_subtoken_strings_with_count.extend((subtoken_counts.get(a, 0), a)
                                  for a in self._alphabet)
      new_subtoken_strings_with_count.sort(reverse=True)

      # Reinitialize to the candidate vocabulary.
      new_subtoken_strings = [subtoken for _, subtoken in new_subtoken_strings_with_count]
      if reserved_tokens:
        # escaped_reserved_tokens = [
        #     _escape_token(native_to_unicode(t), self._alphabet)
        #     for t in reserved_tokens
        # ]
        # new_subtoken_strings = escaped_reserved_tokens + new_subtoken_strings
        new_subtoken_strings = reserved_tokens + new_subtoken_strings

      self._init_subtokens_from_list(new_subtoken_strings)
#       tf.logging.info("vocab_size = %d" % self.vocab_size)
      # print(self.vocab_size)

    self.subtokens_with_counts = new_subtoken_strings_with_count

    # Frequency of "_" is high.
    # So remove from current position and add to the last.
    new_subtoken_strings.remove("_")
    new_subtoken_strings.insert(len(new_subtoken_strings), "_")

    oov_list = []
    for idx, subtoken in enumerate(new_subtoken_strings):
      if subtoken.startswith("_") and subtoken != "_":
        new_subtoken_strings[idx] = subtoken[1:]
      elif subtoken[0] in self._alphabet and subtoken not in reserved_tokens:
        new_subtoken_strings[idx] = "##" + subtoken
      else:
        oov_list.append(subtoken)
    new_subtoken_strings.extend(char for char in self._alphabet
                                    if char not in new_subtoken_strings)

    # print(new_subtoken_strings)
    print("total vocab size : {}, {} seconds elapsed ".format(self.vocab_size, time.time() - start_time))
    # print(oov_list)

    self._init_subtokens_from_list(new_subtoken_strings)

  def _init_subtokens_from_list(self, subtoken_strings, reserved_tokens=None):
    """Initialize token information from a list of subtoken strings.

    Args:
      subtoken_strings: a list of subtokens
      reserved_tokens: List of reserved tokens. We must have `reserved_tokens`
        as None or the empty list, or else the global variable `RESERVED_TOKENS`
        must be a prefix of `reserved_tokens`.

    Raises:
      ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
        is not clear what the space is being reserved for, or when it will be
        filled in.
    """
    if reserved_tokens is None:
      reserved_tokens = []

    if reserved_tokens:
      self._all_subtoken_strings = reserved_tokens + subtoken_strings
    else:
      self._all_subtoken_strings = subtoken_strings

    # we remember the maximum length of any subtoken to avoid having to
    # check arbitrarily long strings.
    self._max_subtoken_len = max([len(s) for s in subtoken_strings])
    self._subtoken_string_to_id = {
        s: i + len(reserved_tokens)
        for i, s in enumerate(subtoken_strings) if s
    }
    # Initialize the cache to empty.
    self._cache_size = 2 ** 20
    self._cache = [(None, None)] * self._cache_size

  def _init_alphabet_from_tokens(self, tokens):
    """Initialize alphabet from an iterable of token or subtoken strings."""
    # Include all characters from all tokens in the alphabet to guarantee that
    # any token can be encoded. Additionally, include all escaping characters.
    self._alphabet = {c for token in tokens for c in token}
    self._alphabet |= _ESCAPE_CHARS
    self._alphabet |= _SPECIAL_CHARS
