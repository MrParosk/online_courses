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

## SQL

Is a programming language used to interact with databases. SQL has two sub-categories:

- Data-definition-language (DDL), it allow us to define the structure of the tables, i.e. CREATE TABLE.
- Data-manipulation-language (DML). it allow us to insert, delete data etc.

## Database design overview

Database design is the process of using the skills we have in order to design a database that doesn't have data integrity issues, no repeating data, old data, anomalies etc. Also we want to split the data up into different tables and describing the relations between them.

For example, we might have a users table and a purchase table, and the relationship between them is the user id.

The process of database design is broken up into three sections:

- Conceptual: thinking about the bigger picture, how to model the relationship between data, brainstorming.
- Logical: more specific, think about the different tables we need and how to relate them, which data-types are needed
- Physical: even more specific, which RDBMS, tables types, which type of server, security etc.

## Data integrity

Data integrity is about having correct data in the database, e.g. no duplicates, broken relationship between tables etc.

Three common data integrity topics are:

- Entity integrity: we want to have unique entities, e.g. not duplicates of the same user.
- Referential integrity: The relationship between tables are correct, e.g. if we have two tables, an user table and a sales table, we can't have a sale row with a non existing user. We can enforce relationships with foreign keys.
- Domain integrity: Acceptable values for a column, e.g. can't have age < 0. This is enforced with database rules, i.e. types (in this case uint). Some databases we can specify general rules, i.e. age => 0.

## Database view

A database view is a searchable object in a database that is defined by a query. Though a view doesn't store data, some refer to a views as "virtual tables", you can query a view like you can a table. A view can combine data from two or more table, using joins, and also just contain a subset of information.
