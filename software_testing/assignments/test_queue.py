# TASK:
#
# Achieve full statement coverage on the Queue class. 
# You will need to:
# 1) Write your test code in the test function.
# 2) Press submit. The grader will tell you if you 
#    fail to cover any specific part of the code.
# 3) Update your test function until you cover the 
#    entire code base.
#
# You can also run your code through a code coverage 
# tool on your local machine if you prefer. This is 
# not necessary, however.
# If you have any questions, please don't hesitate 
# to ask in the forums!

from queue import Queue

def test_enqueue():
	q = Queue(1)
	assert(q.empty())
	assert(q.enqueue(2))
	assert(not q.enqueue(2))
	assert(q.full())

def test_dequeue():
	q = Queue(1)
	assert(q.dequeue() == None)
	assert(q.enqueue(2))
	assert(q.dequeue() == 2)

def test_checkRep():
	q = Queue(2)
	q.checkRep()
	q.enqueue(1)
	q.checkRep()
	q.enqueue(1)
	q.checkRep()
	q.dequeue()
	q.checkRep()
