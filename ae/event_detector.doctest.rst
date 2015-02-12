process_block
=============

>>> from ae.event_detector import process_block
>>> from numpy import array
>>> a = lambda s: array(map(int,s), dtype="i2")

data
****
>>> process_block(None, 0)
Traceback (most recent call last):
  ...
TypeError: data must be array with 'i2' dtype

>>> process_block(array([], dtype="f"), 0)
Traceback (most recent call last):
  ...
TypeError: data must be array with 'i2' dtype

threshold
*********
>>> process_block(a(""), None)
Traceback (most recent call last):
  ...
TypeError: an integer is required

>>> process_block(a(""), 0)
([], None)

hdt
***
>>> process_block(a(""), 0, hdt=None)
Traceback (most recent call last):
  ...
TypeError: an integer is required

>>> process_block(a(""), 0, hdt=0)
([], None)

dead
****
>>> process_block(a(""), 0, dead=None)
Traceback (most recent call last):
  ...
TypeError: an integer is required

>>> process_block(a(""), 0, dead=0)
([], None)

pos
***
>>> process_block(a(""), 0, pos=None)
Traceback (most recent call last):
  ...
TypeError: an integer is required

>>> process_block(a(""), 0, pos=0)
([], None)

>>> process_block(a(""), 0, pos=2**33)
([], None)


event
*****
>>> process_block(a(""), 0, event=0)
Traceback (most recent call last):
  ...
TypeError: event must be None or tuple of two integers

>>> process_block(a(""), 0, event=tuple())
Traceback (most recent call last):
  ...
TypeError: event must be None or tuple of two integers

>>> process_block(a(""), 0, event=(0,0))
([], (0L, 0L))

>>> process_block(a(""), 0, event=None)
([], None)

list
****
>>> process_block(a(""), 0, list=0)
Traceback (most recent call last):
  ...
TypeError: list must be None or list

>>> process_block(a(""), 0, list=None)
([], None)

>>> process_block(a(""), 0, list=[1,2,])
([1, 2], None)

threshold
*********
>>> process_block(a("000010000000"), 1, hdt=3, dead=0)
([], None)

>>> process_block(a("000020000000"), 1, hdt=3, dead=0)
([(4L, 5L)], None)

hdt - joining events
********************
>>> process_block(a("000010000000"), 0, hdt=3, dead=0)
([(4L, 5L)], None)

>>> process_block(a("000011000000"), 0, hdt=3, dead=0)
([(4L, 6L)], None)

>>> process_block(a("000010100000"), 0, hdt=3, dead=0)
([(4L, 7L)], None)

>>> process_block(a("000010010000"), 0, hdt=3, dead=0)
([(4L, 5L), (7L, 8L)], None)

>>> process_block(a("000010010000"), 0, hdt=4, dead=0)
([(4L, 8L)], None)


dead - ignoring events
**********************
>>> process_block(a("000010010000"), 0, hdt=3, dead=3)
([(4L, 5L)], None)

>>> process_block(a("00001101000000000000"), 0, hdt=3, dead=3)
([(4L, 8L)], None)

>>> process_block(a("00001001111110000000"), 0, hdt=3, dead=3)
([(4L, 5L), (10L, 13L)], None)

pos - setting offset
********************
>>> process_block(a("000010000000"), 0, hdt=3, dead=0, pos=1000)
([(1004L, 1005L)], None)

list - appending to supplied
****************************
>>> process_block(a("000010000000"), 0, hdt=3, dead=0, list=[1,2,3])
([1, 2, 3, (4L, 5L)], None)


event - at the end
******************
>>> process_block(a("000010000"), 0, hdt=3, dead=3)
([], (4L, 5L))

>>> process_block(a("0000100000"), 0, hdt=3, dead=3)
([(4L, 5L)], None)

event - restart
***************
>>> process_block(a(""), 0, hdt=3, dead=3, pos=9, event=(4,5))
([], (4L, 5L))

>>> process_block(a(""), 0, hdt=3, dead=3, pos=10, event=(4,5))
([(4L, 5L)], None)

>>> process_block(a("01"), 0, hdt=3, dead=3, pos=9, event=(4,5))
([(4L, 5L)], (10L, 11L))

>>> process_block(a("0100000"), 0, hdt=3, dead=3, pos=9, event=(4,5))
([(4L, 5L), (10L, 11L)], None)


>>> process_block(a("100000"), 0, hdt=3, dead=3, pos=6, event=(4,5))
([(4L, 7L)], None)

>>> process_block(a("0111111100000"), 0, hdt=3, dead=3, pos=6, event=(4,5))
([(4L, 5L), (10L, 14L)], None)

