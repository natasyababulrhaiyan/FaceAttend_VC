import unittest
from unittest.mock import patch, MagicMock
from app import app

class TestLoginWhiteBox(unittest.TestCase):
    def setUp(self):
        # Konfigurasi aplikasi Flask untuk testing
        app.config['TESTING'] = True
        app.secret_key = 'secret123'
        self.client = app.test_client()

    def test_login_get_method(self):
        """
        Path 1: Request metode GET ke /
        Mengekspektasikan render template login.html
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    @patch('app.get_db')
    def test_login_post_valid_admin(self, mock_get_db):
        """
        Path 2: Request POST dengan kredensial valid (Role = admin)
        Mengekspektasikan session dibuat dan redirect ke /admin
        """
        # Mock koneksi dan kursor database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Skenario db mengembalikan data valid sebagai admin
        mock_cursor.fetchone.return_value = {
            'nim': 'admin123',
            'nama': 'Admin Cerdas',
            'role': 'admin'
        }

        # Eksekusi (Kirim form POST)
        response = self.client.post('/', data={
            'nim': 'admin123',
            'password': 'password123'
        })

        # Validasi Assertions
        self.assertEqual(response.status_code, 302) # Status redirect
        self.assertTrue(response.headers['Location'].endswith('/admin')) # Pastikan redirect ke /admin

    @patch('app.get_db')
    def test_login_post_valid_peserta(self, mock_get_db):
        """
        Path 3: Request POST dengan kredensial valid (Role = peserta)
        Mengekspektasikan session dibuat dan redirect ke /peserta
        """
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Skenario db mengembalikan data valid sebagai peserta
        mock_cursor.fetchone.return_value = {
            'nim': '201011400',
            'nama': 'Budi Santoso',
            'role': 'peserta'
        }

        response = self.client.post('/', data={
            'nim': '201011400',
            'password': 'passwordbenar'
        })

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/peserta')) # Pastikan redirect ke /peserta

    @patch('app.get_db')
    def test_login_post_invalid(self, mock_get_db):
        """
        Path 4: Request POST dengan kredensial TIDAK valid
        Mengekspektasikan gagal login dan redirect kembali ke /
        """
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Skenario db TIDAK menemukan user (NIM/Password salah)
        mock_cursor.fetchone.return_value = None

        response = self.client.post('/', data={
            'nim': 'salah',
            'password': 'salah'
        })

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/')) # Pastikan redirect balik ke /

if __name__ == '__main__':
    unittest.main()
