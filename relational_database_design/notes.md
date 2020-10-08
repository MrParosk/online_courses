# Database design

## What is a database?

A database is an organized collection of data. For example in a e-commerce website we will need to store the users and their purchases, this can be done in a database.

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

## Database view

A database view is a searchable object in a database that is defined by a query. Though a view doesn't store data, some refer to a views as "virtual tables", you can query a view like you can a table. A view can combine data from two or more table, using joins, and also just contain a subset of information.

## Database design overview

Database design is the process of using the skills we have in order to design a database that doesn't have data integrity issues, no repeating data, old data, anomalies etc. Also we want to split the data up into different tables and describing the relations between them; a table should be about one entity.

For example, we might have a users table and a purchase table, and the relationship between them is the user id.

The process of database design is broken up into three sections:

- Conceptual: thinking about the bigger picture, how to model the relationship between data, brainstorming.
- Logical: more specific, think about the different tables we need and how to relate them, which data-types are needed.
- Physical: even more specific, which RDBMS, tables types, which type of server, security etc.

## Data integrity

Data integrity is about having correct data in the database, e.g. no duplicates, broken relationship between tables etc.

Three common data integrity topics are:

- Entity integrity: we want to have unique entities, e.g. not duplicates of the same user.
- Referential integrity: The relationship between tables are correct, e.g. if we have two tables, an user table and a sales table, we can't have a sale row with a non existing user. We can enforce relationships with foreign keys.
- Domain integrity: Acceptable values for a column, e.g. can't have age < 0. This is enforced with database rules, i.e. types (in this case uint). Some databases we can specify general rules, i.e. age => 0.

## Atomic values

Atomic values means that we store only one thing in each column. For example, the column "name" with the value "Caleb Daniel Curry" would be split up into first, middle and last name. However we don't want to go to far, i.e. we would not want to split it up into individual characters.

Also we want the values in each column to be singular, e.g. we would store favorite movie instead of favorite movies, and connect users with their favorite movies through relationships.

## Relationships

In a database, multiple tables are connected. Rather than storing everything in a table, we split them into multiple tables. We still need to connect the tables, this is done with relationships.

## One-to-one relationships

One entity has a connection to one other entity. One example is a marriage; one person is married to one person.

### Designing one-to-one relationships

We can usually store one-to-one relationship as attributes, i.e. store it as a column. For example, if we have credit cards (which can only be owned by one person and each person can only have one card) with columns type, amount on the card etc and cardholder with columns first-name, last-name etc we would have columns with card-id and user-id in both tables and connect the entities with foreign keys, i.e. the foreign key in the credit card table would be user-id and in the cardholder the card-id.

However, one can store a one-to-one relationships in one table; depending on the problem.

## One-to-many relationships

One entity has multiple connections to other entities. One example is comments on a website, one user can have multiple comments, but one comment belongs to one user.

### Designing one-to-many relationships

Designing one-to-many is similar to one-to-one, but instead we would only store the user-id in the card table as a foreign key.

### Parent tables and child tables

Primary keys are unique in the table, foreign keys references keys in another table. The table which stores the foreign key is called child table and the table which own the foreign key is called the parent table. The child-table points to the parent table, i.e. Parent <- Child.

## Many-to-many relationships

Many entities has multiple connections to multiple entities. One example is purchased items on a website, multiple products has multiple buyers, but each buyer can buy multiple products.

### Designing many-to-many relationships

The trick to designing many-to-many relationship is to split the data into two tables, each having a one-to-many relationship to a third table, an intermediary / junction table.

In the example above, we would have one buyers table with a primary key (buyer id), one product table with a primary key (product id) and an intermediary table with the foreign keys buyer id and product id.

## Introduction to keys

Keys in databases are used to help with data integrity. (Primary) Keys has the following properties:

- Keys needs to be unique.
- Keys can never change.
- Keys can not be null.

A key is a type of index.

## Lookup-tables

When we want to store categories, e.g. membership type (gold, silver, etc) we can store those types in a table and access them with their ids. Then in the users table, we have a foreign key (the id). Now if we want the change the a name (e.g. gold -> diamond) we simply need to change it at one place;. Lookup tables also helps with data integrity.

## Superkey and candidate key

A superkey is any number of columns that forces a row to be unique. If there isn't, consider adding a row with unique values (e.g. ids). This is usually not defined / discussed in a database.

A candidate key is the least number of columns that forces a row to be unique. Note that we can have multiple candidate keys.

The "workflow" for finding the primary key usually follows:

- Can every row be unique (i.e. superkey)?
- How many columns are needed (candidate key)?
- How many candidate key do I have?

## Primary key and alternative key

From all our candidate keys, choose the "best" one as your primary key, i.e. enforces the constraints for keys and "make sense". The primary key is used to connect tables.

The candidate keys we did not choose are called alternative keys.

## Natural keys and surrogate keys

Natural and surrogate keys are types of primary keys. A natural key is something that you naturally store in the table, e.g. email or username. Surrogate keys is something we come up with, i.e. id number.

In a database, we usually want to use only natural keys or only surrogate keys (i.e. for being consistent and avoid confusion).

Some pros and cons with natural and surrogate keys:

Natural keys:

- Don't need to define any new data.
- Have real-world meaning.
- Don't always find a natural key.

Surrogate keys:

- Can be confusing, since there is typically not and real-world meaning.
- Surrogate keys are usually numbers, which is easy to work with.
- Need to add a column, i.e. extra data to store.

## Foreign keys

Foreign key is a reference to a primary key in another table. It helps connecting tables. We can have multiple foreign keys in a table.

When defining the foreign key, we can put the constraint NOT NULL. It enforces that we need to provide the value. Setting it requires us to specify the relationship.

### Foreign key constraints

Foreign key constraints specifies what will happen to the children when the parent gets updated. This helps with data integrity.

When creating the table, we can specify what will happen when the following things occurs:

- On delete.
- On update.

For each of them, we can specify what will happen:

- Restrict (will throw error).
- Cascade (will do whatever we did to the parent to the child).
- Set null (set the children foreign key to null).

Note that these commands are for MySQL, they might differ between RDBMS.
With set null, the foreign key cannot have the NOT NULL constraint.

## Simple key, composite key and compound key

Simple key - A key composed of one column.
Composite key - A key composed of two or more columns.
Compound key - A key composed of two or more columns and each column is a key themselves. One example where this is used is for intermediate tables used in many-to-many relationship.

Note that some RDBMS uses composite key and compound key interchangeably.
