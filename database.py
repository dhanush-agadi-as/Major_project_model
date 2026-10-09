"""
SQLite Database Manager for VisionMate
Handles face embeddings and item location storage
"""
import sqlite3
import numpy as np
import json
from datetime import datetime
from typing import List, Tuple, Optional
import os
from config import DATABASE_PATH


class VisionMateDB:
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        # Face embeddings table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Item locations table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS item_locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item TEXT NOT NULL,
                location TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item)
            )
        ''')
        
        # Remembered objects table (custom enrolled objects)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS remembered_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                custom_name TEXT NOT NULL UNIQUE,
                yolo_class TEXT NOT NULL,
                visual_features TEXT NOT NULL,
                enrollment_frames INTEGER DEFAULT 1,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Object sightings table (when/where objects were last seen)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id INTEGER NOT NULL,
                location TEXT,
                gps_lat REAL,
                gps_lon REAL,
                context TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (object_id) REFERENCES remembered_objects(id),
                UNIQUE(object_id)
            )
        ''')
        
        self.conn.commit()
    
    def add_face(self, name: str, embedding: np.ndarray) -> int:
        """
        Add a new face to the database
        
        Args:
            name: Person's name
            embedding: 128-d face embedding vector
            
        Returns:
            ID of the inserted record
        """
        embedding_str = json.dumps(embedding.tolist())
        self.cursor.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)",
            (name, embedding_str)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_all_faces(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Retrieve all faces from database
        
        Returns:
            List of (id, name, embedding) tuples
        """
        self.cursor.execute("SELECT id, name, embedding FROM faces")
        results = []
        for row in self.cursor.fetchall():
            face_id, name, embedding_str = row
            embedding = np.array(json.loads(embedding_str))
            results.append((face_id, name, embedding))
        return results
    
    def delete_face(self, face_id: int) -> bool:
        """Delete a face by ID"""
        self.cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def update_item_location(self, item: str, location: str) -> None:
        """
        Update or insert item location
        
        Args:
            item: Item name (e.g., 'keys')
            location: Location where item was found (e.g., 'table')
        """
        self.cursor.execute('''
            INSERT INTO item_locations (item, location, timestamp)
            VALUES (?, ?, ?)
            ON CONFLICT(item) DO UPDATE SET
                location = excluded.location,
                timestamp = excluded.timestamp
        ''', (item, location, datetime.now()))
        self.conn.commit()
    
    def get_item_location(self, item: str) -> Optional[str]:
        """
        Get last known location of an item
        
        Args:
            item: Item name
            
        Returns:
            Location string or None if not found
        """
        self.cursor.execute(
            "SELECT location FROM item_locations WHERE item = ?",
            (item,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def get_all_item_locations(self) -> List[Tuple[str, str, str]]:
        """Get all item locations with timestamps"""
        self.cursor.execute(
            "SELECT item, location, timestamp FROM item_locations"
        )
        return self.cursor.fetchall()
    
    # Remembered Objects Methods
    
    def add_remembered_object(self, custom_name: str, yolo_class: str, 
                             visual_features: dict, enrollment_frames: int = 1) -> int:
        """
        Enroll a custom object with its visual features
        
        Args:
            custom_name: User-friendly name (e.g., "house keys", "work wallet")
            yolo_class: Base YOLO class (e.g., "keys", "bottle")
            visual_features: Dictionary containing feature descriptors
            enrollment_frames: Number of frames used for enrollment
            
        Returns:
            ID of the inserted record
        """
        features_str = json.dumps(visual_features)
        try:
            self.cursor.execute('''
                INSERT INTO remembered_objects (custom_name, yolo_class, visual_features, enrollment_frames)
                VALUES (?, ?, ?, ?)
            ''', (custom_name, yolo_class, features_str, enrollment_frames))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            # Object with this name already exists
            return -1
    
    def get_remembered_object(self, custom_name: str) -> Optional[Tuple[int, str, str, dict, int]]:
        """
        Get a remembered object by custom name
        
        Returns:
            (id, custom_name, yolo_class, visual_features, enrollment_frames) or None
        """
        self.cursor.execute(
            "SELECT id, custom_name, yolo_class, visual_features, enrollment_frames FROM remembered_objects WHERE custom_name = ?",
            (custom_name,)
        )
        result = self.cursor.fetchone()
        if result:
            obj_id, name, yolo_class, features_str, frames = result
            features = json.loads(features_str)
            return (obj_id, name, yolo_class, features, frames)
        return None
    
    def get_remembered_objects_by_class(self, yolo_class: str) -> List[Tuple[int, str, dict]]:
        """
        Get all remembered objects of a specific YOLO class
        
        Returns:
            List of (id, custom_name, visual_features) tuples
        """
        self.cursor.execute(
            "SELECT id, custom_name, visual_features FROM remembered_objects WHERE yolo_class = ?",
            (yolo_class,)
        )
        results = []
        for row in self.cursor.fetchall():
            obj_id, name, features_str = row
            features = json.loads(features_str)
            results.append((obj_id, name, features))
        return results
    
    def get_all_remembered_objects(self) -> List[Tuple[int, str, str]]:
        """Get all remembered objects (id, custom_name, yolo_class)"""
        self.cursor.execute(
            "SELECT id, custom_name, yolo_class FROM remembered_objects"
        )
        return self.cursor.fetchall()
    
    def delete_remembered_object(self, custom_name: str) -> bool:
        """Delete a remembered object and its sightings"""
        # First get the object ID
        obj = self.get_remembered_object(custom_name)
        if not obj:
            return False
        
        obj_id = obj[0]
        
        # Delete sightings first (foreign key constraint)
        self.cursor.execute("DELETE FROM object_sightings WHERE object_id = ?", (obj_id,))
        
        # Delete the object
        self.cursor.execute("DELETE FROM remembered_objects WHERE id = ?", (obj_id,))
        self.conn.commit()
        return True
    
    def add_object_sighting(self, object_id: int, location: str = None,
                           gps_lat: float = None, gps_lon: float = None,
                           context: dict = None, confidence: float = 0.0) -> None:
        """
        Record a sighting of a remembered object
        
        Args:
            object_id: ID of the remembered object
            location: Text description of location
            gps_lat: GPS latitude
            gps_lon: GPS longitude
            context: Dictionary with nearby objects, scene description
            confidence: Match confidence (0-1)
        """
        context_str = json.dumps(context) if context else None
        
        self.cursor.execute('''
            INSERT INTO object_sightings (object_id, location, gps_lat, gps_lon, context, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(object_id) DO UPDATE SET
                location = excluded.location,
                gps_lat = excluded.gps_lat,
                gps_lon = excluded.gps_lon,
                context = excluded.context,
                confidence = excluded.confidence,
                timestamp = excluded.timestamp
        ''', (object_id, location, gps_lat, gps_lon, context_str, confidence, datetime.now()))
        self.conn.commit()
    
    def get_latest_sighting(self, object_id: int) -> Optional[dict]:
        """
        Get the most recent sighting of an object
        
        Returns:
            Dictionary with location, GPS, context, confidence, timestamp
        """
        self.cursor.execute('''
            SELECT location, gps_lat, gps_lon, context, confidence, timestamp
            FROM object_sightings
            WHERE object_id = ?
        ''', (object_id,))
        
        result = self.cursor.fetchone()
        if result:
            location, gps_lat, gps_lon, context_str, confidence, timestamp = result
            context = json.loads(context_str) if context_str else None
            return {
                'location': location,
                'gps_lat': gps_lat,
                'gps_lon': gps_lon,
                'context': context,
                'confidence': confidence,
                'timestamp': timestamp
            }
        return None
    
    def get_sighting_by_name(self, custom_name: str) -> Optional[dict]:
        """Get latest sighting by object's custom name"""
        obj = self.get_remembered_object(custom_name)
        if not obj:
            return None
        return self.get_latest_sighting(obj[0])
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Quick test
    db = VisionMateDB()
    print("[OK] Database initialized successfully")
    print(f"Database location: {db.db_path}")
    db.close()
