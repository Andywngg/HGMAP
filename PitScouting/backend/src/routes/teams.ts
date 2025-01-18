import express from 'express';
import multer from 'multer';
import path from 'path';
import { createTeam, getTeam, getAllTeams, updateTeam } from '../controllers/team.controller';

const router = express.Router();

const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (_req, file, cb) => {
    cb(null, `${Date.now()}${path.extname(file.originalname)}`);
  }
});

const upload = multer({ storage });

router.post('/', upload.single('robotImage'), createTeam);
router.get('/:teamNumber', getTeam);
router.get('/', getAllTeams);
router.put('/:teamNumber', upload.single('robotImage'), updateTeam);

export default router; 