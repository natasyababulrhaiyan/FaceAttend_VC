-- Perbaikan disarankan: biarkan MySQL mengisi id otomatis (lebih baik daripada hitung manual di aplikasi).
-- Sesuaikan tipe kolom jika berbeda (cek dengan: SHOW CREATE TABLE absensi;)

ALTER TABLE absensi
  MODIFY COLUMN id INT NOT NULL AUTO_INCREMENT;
