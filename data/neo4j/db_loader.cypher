// This should be done to cleanup data
MATCH (n) DETACH DELETE (n);

CREATE INDEX author_id_idx FOR (a:Authors) ON (a.author);
CREATE INDEX publisher_id_idx FOR (p:Publishers) ON (p.publisher);
CREATE INDEX year_of_publication_id_idx FOR (y:YearsOfPublication) ON (y.year_of_publication);
CREATE INDEX title_id_idx FOR (t:Titles) ON (t.title);
CREATE INDEX isbn_id_idx FOR (t:Titles) ON (t.isbn);
CREATE INDEX user_id_idx FOR (u:Users) ON (u.user);
CALL db.awaitIndexes();

LOAD CSV WITH HEADERS FROM 'file:///clean_books.csv' AS line
CREATE (:Titles {
        title: line.`Book-Title`,
        isbn: line.ISBN,
        author: line.`Book-Author`,
        year_of_publication: toInteger(line.`Year-Of-Publication`),
        publisher: line.Publisher
	}
);

LOAD CSV WITH HEADERS FROM 'file:///clean_users.csv' AS line
CREATE (:Users {
    user: toInteger(line.`User-ID`),
    location: line.Location,
    age: toInteger(line.Age)
	}
);

LOAD CSV WITH HEADERS FROM 'file:///clean_publishers.csv' AS line
CREATE (:Publishers {
    publisher: line.Publisher
	}
);

LOAD CSV WITH HEADERS FROM 'file:///clean_years_of_publication.csv' AS line
CREATE (:YearsOfPublication {
    year_of_publication: toInteger(line.`Year-Of-Publication`)
	}
);

LOAD CSV WITH HEADERS FROM 'file:///clean_authors.csv' AS line
CREATE (:Authors {
    author: line.`Book-Author`
	}
);

LOAD CSV WITH HEADERS FROM 'file:///clean_ratings.csv' AS line
WITH toInteger(line.`User-ID`) AS UserID, line.ISBN AS ISBN, toInteger(line.`Book-Rating`) AS Rating
WHERE Rating > 0
MATCH (t:Titles {isbn: ISBN})
MATCH (u:Users {user: UserID})
MERGE (t)<-[tu:RATED_BY]-(u)
SET tu.rating = Rating;

LOAD CSV WITH HEADERS FROM 'file:///clean_ratings.csv' AS line
WITH toInteger(line.`User-ID`) AS UserID, line.ISBN AS ISBN, toInteger(line.`Book-Rating`) AS Rating
MATCH (t:Titles {isbn: ISBN})
MATCH (u:Users {user: UserID})
MERGE (t)<-[tu:READ_BY]-(u);

MATCH (t:Titles)
MATCH (p:Publishers)
WHERE t.publisher = p.publisher
MERGE (t)<-[tp:PUBLISHED_BY]-(p);

MATCH (t:Titles)
MATCH (a:Authors)
WHERE t.author = a.author
MERGE (t)<-[tp:WRITTEN_BY]-(a);

MATCH (t:Titles)
MATCH (y:YearsOfPublication)
WHERE t.year_of_publication = y.year_of_publication
MERGE (t)-[tp:WRITTEN_IN_YEAR]->(y);

