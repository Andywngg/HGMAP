import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData extends LoginCredentials {
  name: string;
  teamNumber: number;
}

export interface User {
  id: number;
  name: string;
  email: string;
  teamNumber: number;
}

const authService = {
  login: async (credentials: LoginCredentials): Promise<{ token: string; user: User }> => {
    const response = await axios.post(`${API_URL}/auth/login`, credentials);
    const { token, user } = response.data;
    localStorage.setItem('token', token);
    return { token, user };
  },

  register: async (data: RegisterData): Promise<{ token: string; user: User }> => {
    const response = await axios.post(`${API_URL}/auth/register`, data);
    const { token, user } = response.data;
    localStorage.setItem('token', token);
    return { token, user };
  },

  logout: () => {
    localStorage.removeItem('token');
  },

  getCurrentUser: (): User | null => {
    const token = localStorage.getItem('token');
    if (!token) return null;
    // TODO: Implement token decoding or API call to get user data
    return null;
  },
};

export default authService; 