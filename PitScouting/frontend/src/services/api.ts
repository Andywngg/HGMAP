import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001/api';
const STORAGE_URL = process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:5001';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface TeamData {
  teamNumber: number;
  autoScoreCoral: boolean;
  autoScoreAlgae: boolean;
  autoStartingPosition: string;
  teleopCoralCapability: number;
  teleopAlgaeCapability: number;
  teleopDealgifying: boolean;
  teleopCoralLevels: string[];
  endgameDeepPerformance: number;
  endgameShallowPerformance: number;
  endgameClimbing: boolean;
  robotWidth: number;
  robotLength: number;
  robotHeight: number;
  robotWeight: number;
  drivetrainType: string;
  notes: string;
  updatedAt: string;
  imageUrl: string;
}

export const getImageUrl = (path: string) => {
  if (!path) return '';
  return path.startsWith('http') ? path : `${STORAGE_URL}${path}`;
};

export const teamService = {
  getTeam: async (teamNumber: number): Promise<TeamData> => {
    const response = await api.get(`/teams/${teamNumber}`);
    return response.data;
  },

  updateTeam: async (teamNumber: number, data: Partial<TeamData>): Promise<TeamData> => {
    const response = await api.put(`/teams/${teamNumber}`, data);
    return response.data;
  },

  getAllTeams: async (): Promise<TeamData[]> => {
    const response = await api.get('/teams');
    return response.data;
  },
}; 