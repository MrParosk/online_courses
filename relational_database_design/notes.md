# Database design

## What is a database?

A database is an organized collection of data. For example in a ecommerce website we will need to store the users and their purchases, this can be done in a database.

## Relations and relational databases

Relations is a connection between data. The following terms is used when describing relationships:

- Entity: anything we store data about. For example a person called Caleb.
- Entity type: In the example above, the entity type could be person / user.

- Attribute: the thing we store. For the person, we would have their username admin and password pie123.
- Attribute type: In the example above, the username and password are the attribute types.

The relationship is between entities and attributes.

A relational database is a database where we store data based on this relational model of data. In a table, an attribute is a column, and a row is a collection of attribute for one entity.

## Database Management System (DBMS)

A DBMS is the software that allow us to interact with the database, such as running queries, e.g. find all user with an age above 30 or insert 10 new users to the database or administer the database. A RDBMS is a DBMS where we use the relational data model.

A lot of time, databases and DBMS is refer as the same thing, which is not technically true but in practice are.

Example of RDBMS is MySQL, SQL Server etc.

## Database view
A database view is a searchable object in a database that is defined by a query. Though a view doesn't store data, some refer to a views as "virtual tables", you can query a view like you can a table. A view can combine data from two or more table, using joins, and also just contain a subset of information.
