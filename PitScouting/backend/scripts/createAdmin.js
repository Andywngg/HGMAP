const { Pool } = require('pg');
const bcrypt = require('bcryptjs');

const pool = new Pool({
  user: 'postgres',
  password: 'postgres',
  host: 'localhost',
  database: 'scouting_app',
  port: 5432
});

async function createAdmin() {
  try {
    const hashedPassword = await bcrypt.hash('otisit!!!', 10);
    const now = new Date().toISOString();
    
    const query = `
      INSERT INTO users (name, email, password, "teamNumber", "createdAt", "updatedAt")
      VALUES ($1, $2, $3, $4, $5, $6)
    `;
    
    await pool.query(query, [
      'Team 1334 Admin',
      '1334admin@gmail.com',
      hashedPassword,
      1334,
      now,
      now
    ]);
    
    console.log('Admin user created successfully');
  } catch (error) {
    console.error('Error creating admin:', error);
  } finally {
    pool.end();
  }
}

createAdmin(); 