import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import path from 'path';

let db: any = null;

export async function getDb() {
  if (!db) {
    db = await open({
      filename: path.join(process.cwd(), 'xgboost_models.db'),
      driver: sqlite3.Database,
    });

    await db.exec(`
      CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        csv_filename TEXT NOT NULL,
        target_column TEXT NOT NULL,
        feature_columns TEXT NOT NULL,
        model_params TEXT NOT NULL,
        auc_score REAL,
        accuracy REAL,
        feature_importance TEXT,
        confusion_matrix TEXT,
        status TEXT DEFAULT 'training'
      );

      CREATE TABLE IF NOT EXISTS training_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_id) REFERENCES models(id)
      );
    `);
  }
  return db;
}