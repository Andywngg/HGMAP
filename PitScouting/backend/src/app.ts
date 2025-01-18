import express from 'express';
import cors from 'cors';
import teamRoutes from './routes/teams';

const app = express();

app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));
app.use('/api/teams', teamRoutes);

export default app; 