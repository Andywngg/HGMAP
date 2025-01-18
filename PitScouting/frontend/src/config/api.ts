const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001/api';
const BASE_URL = API_URL.replace('/api', '');

// Helper to get the base URL for the current environment
const getBaseUrl = () => {
  // If we have an explicit API URL from environment, use it
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL.replace('/api', '');
  }
  
  // For production, use the current window location
  if (process.env.NODE_ENV === 'production') {
    const url = new URL(window.location.href);
    return `${url.protocol}//${url.host}`;
  }
  
  // Default to localhost for development
  return 'http://localhost:5001';
};

const getAuthHeader = () => {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

export const endpoints = {
  auth: {
    login: `${getBaseUrl()}/api/auth/login`,
    register: `${getBaseUrl()}/api/auth/register`,
  },
  teams: {
    list: `${getBaseUrl()}/api/teams`,
    create: `${getBaseUrl()}/api/teams`,
    get: (teamNumber: string | number) => `${getBaseUrl()}/api/teams/${teamNumber}`,
    getConfig: () => getAuthHeader(),
  },
  uploads: {
    getUrl: (path: string) => {
      if (!path) return '';
      if (path.startsWith('http')) return path;
      return `${getBaseUrl()}/storage${path.replace('/uploads', '')}`;
    },
  },
};

export default endpoints; 