# TASK:
#
# This is the SplayTree code we saw earlier in the 
# unit. We didn't achieve full statement coverage 
# during the unit, but we will now!
# Your task is to achieve full statement coverage 
# on the SplayTree class. 
# 
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

from splay_tree import SplayTree, Node

def test_node():
	n = Node(1)
	n_ = Node(1)
	assert(n.equals(n_))

def test_insert():
	t = SplayTree()
	t.insert(1)
	t.insert(1)

def test_remove():
	t = SplayTree()
	t.remove(1)
	t.insert(1)
	t.insert(2)
	t.remove(2)
	t.remove(1)

def test_findMin():
	t = SplayTree()
	assert(t.findMin() == None)
	t.insert(1)
	t.insert(2)
	assert(t.findMin() == 1)

def test_findMax():
	t = SplayTree()
	assert(t.findMax() == None)
	t.insert(1)
	t.insert(-1)
	assert(t.findMax() == 1)

def test_find():
	t = SplayTree()
	assert(t.find(1) == None)
	t.insert(1)
	assert(t.find(1) == 1)
	assert(t.find(10) == None)

def test_isEmpty():
	t = SplayTree()
	assert(t.isEmpty())

def test_splay():
	t = SplayTree()
	t.insert(1)
	t.insert(2)
	t.insert(-1)
	t.insert(10)
	t.insert(5)
	t.splay(1)
	t.splay(1)

	t_ = SplayTree()
	t_.insert(2)
	t_.insert(1)
	t_.insert(10)
	t_.splay(2)
