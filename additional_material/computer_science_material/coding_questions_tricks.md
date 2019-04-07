# Coding questions tricks

## Linked list

The "runner" (or second pointer) technique is used in many linked list problems. The runner technique means that you iterate through the linked list with two pointers simultaneously, with one ahead of the
other. The "fast" node might be ahead by a fixed amount, or it might be hopping multiple nodes for each one node that the "slow" node iterates through.

For example, suppose you had a linked list a1 => a2 => ... => an  => b1  => b2 => ... => bn and you wanted to rearrange it into a1 => b1 => a2 => b2 => ... => an => bn. You do not know the length of the linked list (but you do know that the length is an even number). You could have one pointer pl (the fast pointer) move every two elements for every one move that p2 makes. When pl hits the end of the linked list, p2 will be at the midpoint. Then, move pl back to the front and begin "weaving" the elements. On each iteration, p2 selects an element and inserts it after pl.

## Stack

One case where stacks are often useful is in certain recursive algorithms. Sometimes you need to push temporary data onto a stack as you recurse, but then remove them as you backtrack (for example, because the recursive check failed). A stack offers an intuitive way to do this. A stack can also be used to implement a recursive algorithm iteratively.

## Queue

One place where queues are often used is in breadth-first search or in implementing a cache. In breadth-first search, for example, we used a queue to store a list of the nodes that we need to process.
Each time we process a node, we add its adjacent nodes to the back of the queue. This allows us to process nodes in the order in which they are viewed.

## Amortized Time

A dynamically resizing array allows you to have the benefits of an array while offering flexibility in size. You won't run out of space in the array list since its capacity will grow as you insert elements. An array list is implemented with an array. When the array hits capacity, the array list class will create a new array with double the capacity and copy all the elements over to the new array. How do you describe the runtime of insertion? The array could be full. If the array contains N elements, then inserting a new element will take O(N) time. You will have to create a new array of size 2N and then copy N elements over. This insertion will take O (N) time. However, we also know that this doesn't happen very often. The vast majority of the time insertion will be in O(l) time. Therefore the amortized time is O(1).

## DFS and BFS

The same graph algorithms that are used on adjacency lists (breadth-first search, etc.) can be performed with adjacency matrices, but they may be somewhat less efficient. In the adjacency list representation, you
can easily iterate through the neighbors of a node. In the adjacency matrix representation, you will need to iterate through all the nodes to identify a node's neighbors.

Breadth-first search and depth-first search tend to be used in different scenarios. DFS is often preferred if we want to visit every node in the graph. Both will work just fine, but depth-first search is a bit simpler.
However, if we want to find the shortest path (or just any path) between two nodes, BFS is generally better.

The time complexity of both BFS and DFS will be O(V + E), where V is the number of vertices, and E is the number of edges. This again depends on the data structure that we user to represent the graph. If it is an adjacency matrix, it will be O(V^2). If we use an adjacency list, it will be O(V+E).

## Cycles in graphs

To detect cycles in directed graphs, one could use a DFS with the following color-schema:

- Mark current node as grey.
- Recursively, go to each neighbor.
- If we detect any neighbors that are grey, we have a cycle.
- Mark current node as black.

## Recursion

Drawing the recursive calls as a tree is a great way to figure out the runtime of a recursive algorithm.

Recursive algorithms can be very space inefficient. Each recursive call adds a new layer to the stack, which means that if your algorithm recurses to a depth of n, it uses at least O(n) memory.