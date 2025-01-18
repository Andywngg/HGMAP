const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

export const endpoints = {
  auth: {
    login: `${API_URL}/api/auth/login`,
    register: `${API_URL}/api/auth/register`,
  },
  teams: {
    list: `${API_URL}/api/teams`,
    create: `${API_URL}/api/teams`,
    get: (teamNumber: string | number) => `${API_URL}/api/teams/${teamNumber}`,
  },
  uploads: {
    getUrl: (path: string) => `${API_URL}${path}`,
  },
};

export default endpoints; 