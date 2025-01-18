import axios from 'axios';
import endpoints from '../config/api';

const api = axios.create({
  baseURL: endpoints.teams.list.replace('/api/teams', ''),
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

export interface TeamData {
  teamNumber: number;
  autoScoreCoral: boolean;
  autoScoreAlgae: boolean;
  mustStartSpecificPosition: boolean;
  autoStartingPosition: string;
  teleopDealgifying: boolean;
  teleopPreference: string;
  scoringPreference: string;
  coralLevels: string[];
  endgameType: string;
  robotWidth: number;
  robotLength: number;
  robotHeight: number;
  robotWeight: number;
  drivetrainType: string;
  imageUrl: string | null;
}

export const getImageUrl = (path: string) => {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${endpoints.teams.list.replace('/api/teams', '')}/storage${path.replace('/uploads', '')}`;
};

export const teamService = {
  getTeam: async (teamNumber: number): Promise<TeamData> => {
    const response = await api.get(`/api/teams/${teamNumber}`);
    return response.data;
  },

  updateTeam: async (teamNumber: number, data: Partial<TeamData>): Promise<TeamData> => {
    const response = await api.put(`/api/teams/${teamNumber}`, data);
    return response.data;
  },

  getAllTeams: async (): Promise<TeamData[]> => {
    const response = await api.get('/api/teams');
    return response.data;
  },
}; 