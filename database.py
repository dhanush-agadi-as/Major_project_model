"""
SQLite Database Manager for VisionMate
Handles face embeddings and item location storage with thread-safe access.
"""
import sqlite3
import numpy as np
import json
import threading
from datetime import datetime
from typing import List, Tuple, Optional
import os
import logging
from config import DATABASE_PATH

logger = logging.getLogger(__name__)


def is_legacy_62720d(features: dict) -> bool:
    """Detect if visual features contain legacy 62720D embeddings instead of 1280D."""
    if not isinstance(features, dict):
        return False
    if features.get('embedding_size') == 62720:
        return True
    de = features.get('deep_embedding')
    if de is not None and len(de) == 62720:
        return True
    des = features.get('deep_embeddings')
    if des and isinstance(des, list) and len(des) > 0:
        if isinstance(des[0], (list, np.ndarray)) and len(des[0]) == 62720:
            return True
    return False


class VisionMateDB:
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        self._lock = threading.RLock()
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = None
        self.is_closed = False
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        with self._lock:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    @property
    def cursor(self):
        """
        Provide a fresh cursor for backward compatibility with external code/tests.
        Internal methods instantiate their own local cursor per call.
        """
        with self._lock:
            if self.conn and not self.is_closed:
                return self.conn.cursor()
            return None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        with self._lock:
            cur = self.conn.cursor()
            try:
                # Face embeddings table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Item locations table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS item_locations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item TEXT NOT NULL,
                        location TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(item)
                    )
                ''')
                
                # Remembered objects table (custom enrolled objects)
                cur.execute('''
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
                cur.execute('''
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
            finally:
                cur.close()
    
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
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO faces (name, embedding) VALUES (?, ?)",
                    (name, embedding_str)
                )
                self.conn.commit()
                return cur.lastrowid
            finally:
                cur.close()
    
    def get_all_faces(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Retrieve all faces from database
        
        Returns:
            List of (id, name, embedding) tuples
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("SELECT id, name, embedding FROM faces")
                results = []
                for row in cur.fetchall():
                    face_id, name, embedding_str = row
                    embedding = np.array(json.loads(embedding_str))
                    results.append((face_id, name, embedding))
                return results
            finally:
                cur.close()
    
    def delete_face(self, face_id: int) -> bool:
        """Delete a face by ID"""
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("DELETE FROM faces WHERE id = ?", (face_id,))
                self.conn.commit()
                return cur.rowcount > 0
            finally:
                cur.close()

    def delete_face_by_name(self, name: str) -> bool:
        """Delete all faces matching name (case-insensitive)"""
        if not name:
            return False
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("DELETE FROM faces WHERE LOWER(name) = LOWER(?)", (name.strip(),))
                self.conn.commit()
                return cur.rowcount > 0
            finally:
                cur.close()

    def rename_face(self, old_name: str, new_name: str) -> bool:
        """Rename faces from old_name to new_name (case-insensitive)"""
        if not old_name or not new_name:
            return False
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("UPDATE faces SET name = ? WHERE LOWER(name) = LOWER(?)", (new_name.strip(), old_name.strip()))
                self.conn.commit()
                return cur.rowcount > 0
            finally:
                cur.close()
    
    def update_item_location(self, item: str, location: str) -> None:
        """
        Update or insert item location
        
        Args:
            item: Item name (e.g., 'keys')
            location: Location where item was found (e.g., 'table')
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute('''
                    INSERT INTO item_locations (item, location, timestamp)
                    VALUES (?, ?, ?)
                    ON CONFLICT(item) DO UPDATE SET
                        location = excluded.location,
                        timestamp = excluded.timestamp
                ''', (item, location, datetime.now()))
                self.conn.commit()
            finally:
                cur.close()
    
    def get_item_location(self, item: str) -> Optional[str]:
        """
        Get last known location of an item
        
        Args:
            item: Item name
            
        Returns:
            Location string or None if not found
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "SELECT location FROM item_locations WHERE item = ?",
                    (item,)
                )
                result = cur.fetchone()
                return result[0] if result else None
            finally:
                cur.close()
    
    def get_all_item_locations(self) -> List[Tuple[str, str, str]]:
        """Get all item locations with timestamps"""
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "SELECT item, location, timestamp FROM item_locations"
                )
                return cur.fetchall()
            finally:
                cur.close()
    
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
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute('''
                    INSERT INTO remembered_objects (custom_name, yolo_class, visual_features, enrollment_frames)
                    VALUES (?, ?, ?, ?)
                ''', (custom_name, yolo_class, features_str, enrollment_frames))
                self.conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                # Object with this name already exists
                return -1
            finally:
                cur.close()
    
    def get_remembered_object(self, custom_name: str) -> Optional[Tuple[int, str, str, dict, int]]:
        """
        Get a remembered object by custom name
        
        Returns:
            (id, custom_name, yolo_class, visual_features, enrollment_frames) or None
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "SELECT id, custom_name, yolo_class, visual_features, enrollment_frames FROM remembered_objects WHERE custom_name = ?",
                    (custom_name,)
                )
                result = cur.fetchone()
                if result:
                    obj_id, name, yolo_class, features_str, frames = result
                    features = json.loads(features_str)
                    if is_legacy_62720d(features):
                        logger.warning(
                            f"Old 62720D row detected for '{name}'. "
                            f"Invalid embedding dimension (62720D); ignoring object. Please re-enroll."
                        )
                        return None
                    return (obj_id, name, yolo_class, features, frames)
                return None
            finally:
                cur.close()
    
    def get_remembered_objects_by_class(self, yolo_class: str) -> List[Tuple[int, str, dict]]:
        """
        Get all remembered objects of a specific YOLO class
        
        Returns:
            List of (id, custom_name, visual_features) tuples
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "SELECT id, custom_name, visual_features FROM remembered_objects WHERE yolo_class = ?",
                    (yolo_class,)
                )
                results = []
                for row in cur.fetchall():
                    obj_id, name, features_str = row
                    features = json.loads(features_str)
                    if is_legacy_62720d(features):
                        logger.warning(
                            f"Old 62720D row detected for '{name}'. "
                            f"Invalid embedding dimension (62720D); ignoring object. Please re-enroll."
                        )
                        continue
                    results.append((obj_id, name, features))
                return results
            finally:
                cur.close()
    
    def get_all_remembered_objects(self) -> List[Tuple[int, str, str]]:
        """Get all remembered objects (id, custom_name, yolo_class)"""
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "SELECT id, custom_name, yolo_class FROM remembered_objects"
                )
                return cur.fetchall()
            finally:
                cur.close()
    
    def delete_remembered_object(self, custom_name: str) -> bool:
        """Delete a remembered object and its sightings"""
        with self._lock:
            # First get the object ID
            obj = self.get_remembered_object(custom_name)
            if not obj:
                return False
            
            obj_id = obj[0]
            cur = self.conn.cursor()
            try:
                # Delete sightings first (foreign key constraint)
                cur.execute("DELETE FROM object_sightings WHERE object_id = ?", (obj_id,))
                # Delete the object
                cur.execute("DELETE FROM remembered_objects WHERE id = ?", (obj_id,))
                self.conn.commit()
                return True
            finally:
                cur.close()
    
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
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute('''
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
            finally:
                cur.close()
    
    def get_latest_sighting(self, object_id: int) -> Optional[dict]:
        """
        Get the most recent sighting of an object
        
        Returns:
            Dictionary with location, GPS, context, confidence, timestamp
        """
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute('''
                    SELECT location, gps_lat, gps_lon, context, confidence, timestamp
                    FROM object_sightings
                    WHERE object_id = ?
                ''', (object_id,))
                
                result = cur.fetchone()
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
            finally:
                cur.close()
    
    def get_sighting_by_name(self, custom_name: str) -> Optional[dict]:
        """Get latest sighting by object's custom name"""
        with self._lock:
            obj = self.get_remembered_object(custom_name)
            if not obj:
                return None
            return self.get_latest_sighting(obj[0])
    
    def close(self):
        """Close database connection"""
        with self._lock:
            self.is_closed = True
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
    
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
