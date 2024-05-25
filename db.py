import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2
from mistralai.client import MistralClient


class MistralAPI:
    def __init__(self):
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    def embed(self, txt):
        """ 
        Embedd a single string
        """
        embedding = self.client.embeddings("mistral-embed", txt)
        return embedding.data[0].embedding
    
class Database:
    def __init__(self):
        # Connect to the Postgres database
        self.connection_string = os.getenv('DATABASE_URL')
        self.conn = psycopg2.connect(self.connection_string)
        self.cur = self.conn.cursor()
    
    def closeConn(self):
        self.conn.close()
        self.cur.close()
    
    def testConnection(self):
        # Execute SQL commands to retrieve the current time and version from PostgreSQL
        self.cur.execute('SELECT NOW();')
        time = self.cur.fetchone()[0]
        self.cur.execute('SELECT version();')
        version = self.cur.fetchone()[0]

        # Print the results
        print('Current time:', time)
        print('PostgreSQL version:', version)

    def insertImage(self, caption, image_filename, video_filename, embedding):
        # SQL query to insert or update data into the table 'lechaton'
        query = """
        INSERT INTO lechaton (video_filename, image_filename, caption, embedding) 
        VALUES (%s, %s, %s, %s)
        """
        # Execute the query
        self.cur.execute(query, (video_filename, image_filename, caption, embedding))
        # Commit the changes to the database
        self.conn.commit()
        # maybe add readall here?

    def search(self, embedding, k):
        # Transform the embedding list to a string format for SQL query compatibility
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        query = """
            SELECT * FROM lechaton ORDER BY embedding <=> (%s) LIMIT (%s);
        """
        # Execute the query
        self.cur.execute(query, (embedding_str, k))
        # Commit the changes to the database
        self.conn.commit()

        rows = self.cur.fetchall()
        # Return the fetched rows
        return rows

class VectorStore:
    def __init__(self):
        self.db = Database()
        self.mistral_client = MistralAPI()

    def insert(self, caption, image_filename, video_filename):
        embedding = self.mistral_client.embed(caption)
        self.db.insertImage(caption, image_filename, video_filename, embedding)

    def search(self, text, k):
        embedding = self.mistral_client.embed(text)
        rows = self.db.search(embedding, k)
        return rows
    
    def close(self):
        self.db.closeConn()

def test_add_to_vs():
    vs = VectorStore()

    vs = VectorStore()
    mistral_folder = 'Mistral'
    files = os.listdir(mistral_folder)[:5]  # Get the first 5 files from the Mistral folder
    captions = [
        "A serene landscape",
        "A bustling cityscape",
        "A quiet moment",
        "The thrill of adventure",
        "A night under the stars"
    ]
    
    for i, file in enumerate(files):
        if i > 5:
            break
        if file.endswith('.jpg'):
            image_filename = os.path.join(mistral_folder, file)
            video_filename = 'demo'
            caption = captions[i]
            vs.insert(caption, image_filename, video_filename)

def test_search_vs():
    vs = VectorStore()
    rows = vs.search("A small group of humans", 2)
    print(rows)

if __name__ == "__main__":
    #test_add_to_vs()
    test_search_vs()
    pass