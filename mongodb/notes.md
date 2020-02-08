# MongoDB - The Complete Developer's Guide

## MongoDB basics
- MongoDB is a NoSQL database.
- NoSQL is "all about flexibility".
- In NoSQL we don't normalizing the data (i.e. in SQL we store the data at multiple places / tables).
- Can lead to messy data, however its up to the developer to make sure this doesn't happen.
- In MongoDB, each database consist of multiple collections (collections ≈ tables in SQL).
- In each collections there are multiple documents (documents ≈ rows in SQL).

<img src="./images/mongo_db_example.png" width="500"/>

## MongoDB backend structure

- MongoDB is popular for heavy read & write applications.
- As an application developer, we interact with the MongoDB server, which in turn is responsible for communicating with the storage engine (the default is WiredTiger). The storage engine then accesses data / files from the file-system.
- We can communicate with the MongoDB server either from the drivers or from the MongoDB shell.

<img src="./images/example_application_layout.png" width="500"/>

## Data format

- MongoDB stores data in BSON (binary version of JSON).
- However when inserting documents, we use JSON, which then gets converted to BSON.

<img src="./images/bson.png" width="500"/>

## CRUD operations
- CRUD stands for Create, Read, Update and Delete and are command commands in MongoDB.

<img src="./images/crud_commands.png" width="500"/>

## Cursors
- When using fetch-multiple-data-command such as collection.find() MongoDB doesn't return the data but a cursor object.
- From the cursor object we can fetch the data by .toArray() or similar command.
- Think of them as similar to Python's iterators.

<img src="./images/cursors.png" width="500"/>

<img src="./images/understand_cursors.png" width="500"/>

## Projections
- Sometimes we only want to return some fields of the documents.
- We could fetch all documents and them only keep the desired fields, however this would require more bandwidth.
- Instead we can use projections, which allow us to do these operations in the database and only return the stripped down version of the data.
- The command follow .find(query, {key: value}), where key is the field/s we want to keep / remove.
- Value 1 means keep and 0 means drop. Note that all fields have the default value 0, except the _id field which has the default value 1.
- E.g. .find({}, {name: 1, _id: 0}) would only return the name field.

<img src="./images/projections.png" width="500"/>

## Embedded Documents:
- Field in documents can be another document, i.e. nested documents.
- Note that the nested document can have their own id's field.

<img src="./images/embedded_documents.png" width="500"/>

## Schemas

- In NoSQL we can store different types of data in the same collection.
- However, it is usually a good idea to separate the data into different collections based on their types (i.e. the extra data example below).
- Yet, requiring the documents to have the exact same fields defeats the purpose of NoSQL.

<img src="./images/schema_tradeoffs.png" width="500"/>

## Schema-validation
- In MongoDB we can have schema-validation, i.e. inserted / updated documents needs to have a specific structure with specific data-types in order to get inserted.
- We can set the validation level to "strict" (all inserts & updates are checked) or "moderate" (all inserts & updates to correct document).
- We can set the validation action to "error" (throw error and deny insert/update) or "warn" (log warning but proceed).
- It is easiest to insert the validation schema when creating the collection (see below for example).
- However, we can also update an existing schema with db.runCommand({"CollMod": "posts", validator: {...}}).

```javascript
db.createCollection('posts', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['title', 'text', 'creator', 'comments'],
      properties: {
        title: {
          bsonType: 'string',
          description: 'must be a string and is required'
        },
        text: {
          bsonType: 'string',
          description: 'must be a string and is required'
        },
        creator: {
          bsonType: 'objectId',
          description: 'must be an objectid and is required'
        },
        comments: {
          bsonType: 'array',
          description: 'must be an array and is required',
          items: {
            bsonType: 'object',
            required: ['text', 'author'],
            properties: {
              text: {
                bsonType: 'string',
                description: 'must be a string and is required'
              },
              author: {
                bsonType: 'objectId',
                description: 'must be an objectid and is required'
              }
            }
          }
        }
      }
    }
  }
});
```

## Data types
- MongoDB can handle multiple data-types:
    - Text: "Max".
    - Boolean: true
    - Number:
        - Integer (int32): 32
        - NumberLong (int64): 2313123123
        - NumberDecimal (float64): 3.556
    - ObjectId: ObjectId("34asd4444fghh")
    - ISODate: ISODate("2018-09-09")
    - Timestamp: Timestamp(11421532)
    - Embedded Document: {"a": {...}}
    - Array: {b: [...]}
- Note that if we insert a number from the shell it will be of type float64. This is because the shell is based on javascript.
- This might differ for driver in other languages, e.g. Java.

## References versus embedded documents
- For some applications we need relationships. For example if we have created an online forum, we will have users and posts, and there will be relationships between the users and posts.
- We have two options: references or embedded documents.
- For references, we keep an reference (e.g. id-field) s.t. we can lookup the required document fast.
- For embedded documents, we embed the sub-document within our "main" document.
- There are three types of relationships commonly discussed, one-to-one, one-to-many, many-to-many.
- There are some rule-of-thumb for when to choose which type (i.e. references or embedded), however the choice is context dependent.
- One must always think about how the data is fetched, how often its fetched, does the data change a lot, gets remove etc.

### References
- Split data across collections.
- Great for related but shared data as well as for data which is used in relations and standalone.
- Allows you to overcome nesting and size limits (by creating new documents).
- Can use MongoDB's $lookup command for accessing the reference-data.

#### Reference example
```json
{
    "_id": "uid",
    "username": "user",
    "posts": ["id_1", "id_2"]
}

{
    "_id": "id_1",
    "title": "post 1",
    "text": "topic 1"
},
{
    "_id": "id_2",
    "title": "post 2",
    "text": "topic 2"
}
```

### Embedded documents
- Group data together logically.
- Great for data that belongs together and is not really overlapping with other data.
- Avoid super-deep nesting (100+ levels) or extremely long arrays (16 mb size limit per document).

#### Embedded example
```json
{
    "_id": "uid",
    "username": "user",
    "posts": [
        {"title": "post 1", "text": "topic 1"},
        {"title": "post 2", "text": "topic 2"}
    ]
}
```

## One-to-one
- One-to-one often uses the embedded document structure, however depends on the application.
- Example of one-to-one relationship:

<img src="./images/one_one.png" width="500"/>

## One-to-many
- One-to-many often uses the embedded document structure, however depends on the application.
- Example of one-to-many relationship:

<img src="./images/one_many.png" width="500"/>


## Many-to-many
- Many-to-many often uses the references structure, however depends on the application.
- Example of many-to-many relationship:

<img src="./images/many_many.png" width="500"/>
