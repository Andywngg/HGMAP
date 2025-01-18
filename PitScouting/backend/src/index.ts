import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { sequelize } from './models';
import authRoutes from './routes/auth';
import teamRoutes from './routes/teams';

const app = express();
const port = process.env.PORT || 5001;
const env = process.env.NODE_ENV || 'development';
const corsOrigin = env === 'production' 
  ? process.env.FRONTEND_URL || 'https://your-frontend-url.vercel.app'
  : 'http://localhost:3000';

// CORS configuration
app.use(cors({
  origin: corsOrigin,
  credentials: true
}));

app.use(express.json());

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Serve static files from the uploads directory
app.use('/storage', express.static(uploadsDir));

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/teams', teamRoutes);

// Root route
app.get('/', (_req, res) => {
  res.json({ 
    message: 'Scouting App API is running',
    environment: env,
    corsOrigin
  });
});

// Database connection and server start
sequelize.authenticate()
  .then(() => {
    console.log('Database connected successfully');
    app.listen(port, () => {
      console.log(`Server running on port ${port}`);
      console.log(`Environment: ${env}`);
      console.log(`CORS origin: ${corsOrigin}`);
      console.log('API endpoints:');
      console.log('- POST /api/auth/login');
      console.log('- POST /api/auth/register');
      console.log('- GET /api/teams');
      console.log('- POST /api/teams');
      console.log('- GET /api/teams/:teamNumber');
    });
  })
  .catch((error) => {
    console.error('Unable to connect to the database:', error);
  });

export default app; 