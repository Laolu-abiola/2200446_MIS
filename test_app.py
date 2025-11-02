# Contents of /Surname-MatricNumber/tests/test_app.py

import unittest
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload Image', response.data)

    def test_emotion_detection(self):
        # Assuming there's a route for emotion detection
        response = self.app.post('/detect-emotion', data={'image': (bytes(), 'test.jpg')})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Emotion Detected', response.data)

if __name__ == '__main__':
    unittest.main()