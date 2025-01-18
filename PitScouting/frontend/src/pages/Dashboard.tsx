import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Table from '../components/Table';
import Button from '../components/Button';
import axios from 'axios';
import toast from 'react-hot-toast';
import endpoints from '../config/api';

interface ScoutingData {
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

interface Column {
  key: string;
  header: string;
  render?: (value: any, row?: ScoutingData) => React.ReactNode;
}

const Dashboard = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDrivetrain, setFilterDrivetrain] = useState('');
  const [filterEndgameType, setFilterEndgameType] = useState('');
  const [filterAutoPosition, setFilterAutoPosition] = useState('');
  const [filterTeleopPreference, setFilterTeleopPreference] = useState('');
  const [teams, setTeams] = useState<ScoutingData[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    teamNumber: '',
    drivetrain: '',
    autoCapabilities: '',
    endgameType: '',
    teleopCapability: ''
  });

  useEffect(() => {
    const fetchTeams = async () => {
      try {
        const response = await axios.get(endpoints.teams.list);
        setTeams(response.data);
      } catch (error: any) {
        console.error('Error fetching teams:', error);
        toast.error(error.response?.data?.error || 'Failed to load teams');
      } finally {
        setLoading(false);
      }
    };

    fetchTeams();
  }, []);

  const columns: Column[] = [
    { key: 'teamNumber', header: 'Team Number' },
    {
      key: 'autoScoreCoral',
      header: 'Auto Coral',
      render: (value: boolean) => (value ? '✓' : '✗'),
    },
    {
      key: 'autoScoreAlgae',
      header: 'Auto Algae',
      render: (value: boolean) => (value ? '✓' : '✗'),
    },
    {
      key: 'autoStartingPosition',
      header: 'Starting Position',
      render: (value: string, row?: ScoutingData) => 
        row?.mustStartSpecificPosition ? value : 'No specific position',
    },
    {
      key: 'teleopPreference',
      header: 'Teleop Scoring Type',
      render: (value: string) => {
        switch(value) {
          case 'coral': return 'Coral Only';
          case 'algae': return 'Algae Only';
          case 'both': return 'Both';
          default: return '-';
        }
      },
    },
    {
      key: 'scoringPreference',
      header: 'Scoring Preference',
      render: (value: string) => {
        switch(value) {
          case 'betterCoral': return 'Better at Coral';
          case 'betterAlgae': return 'Better at Algae';
          case 'equal': return 'Equal';
          default: return '-';
        }
      },
    },
    {
      key: 'teleopDealgifying',
      header: 'Can Dealgify',
      render: (value: boolean) => (value ? '✓' : '✗'),
    },
    {
      key: 'endgameType',
      header: 'Endgame',
      render: (value: string) => {
        switch(value) {
          case 'deep': return 'Deep Climb';
          case 'shallow': return 'Shallow Climb';
          case 'none': return 'No Climb';
          default: return '-';
        }
      },
    },
    { key: 'drivetrainType', header: 'Drivetrain' },
    {
      key: 'imageUrl',
      header: 'Robot Image',
      render: (value: string) => value ? (
        <img 
          src={endpoints.uploads.getUrl(value)} 
          alt="Robot" 
          className="w-16 h-16 object-cover rounded cursor-pointer"
          onClick={(e) => {
            e.stopPropagation();
            window.open(endpoints.uploads.getUrl(value), '_blank');
          }}
        />
      ) : null,
    },
  ];

  const filteredTeams = teams?.filter(team => {
    const matchesSearch = !searchTerm || 
      team.teamNumber.toString().includes(searchTerm);
    const matchesDrivetrain = !filterDrivetrain || 
      team.drivetrainType?.toLowerCase().includes(filterDrivetrain.toLowerCase());
    const matchesEndgame = !filterEndgameType || 
      team.endgameType === filterEndgameType;
    const matchesAutoScoring = !filterAutoPosition || 
      (filterAutoPosition === 'coral' && team.autoScoreCoral) ||
      (filterAutoPosition === 'algae' && team.autoScoreAlgae) ||
      (filterAutoPosition === 'both' && team.autoScoreCoral && team.autoScoreAlgae);
    const matchesTeleopPreference = !filterTeleopPreference || 
      team.teleopPreference === filterTeleopPreference;
    
    return matchesSearch && matchesDrivetrain && matchesEndgame && 
           matchesAutoScoring && matchesTeleopPreference;
  });

  const handleExport = () => {
    const headers = columns.map((col) => col.header).join(',');
    const rows = filteredTeams
      .map((row) => columns.map((col) => row[col.key as keyof ScoutingData]).join(','))
      .join('\n');
    const csv = `${headers}\n${rows}`;

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scouting-data.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex justify-center items-center">
        <div className="text-xl text-gray-700">Loading teams...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-white">Scouting Dashboard</h1>
          <div className="flex gap-4">
            <Button onClick={() => navigate('/scout')} className="bg-red-600 hover:bg-red-700 text-white">
              New Scout
            </Button>
            <Button onClick={handleExport} className="bg-gray-600 hover:bg-gray-700 text-white">
              Export CSV
            </Button>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg shadow mb-8">
          <div className="p-6 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">
                  Search Team Number
                </label>
                <input
                  type="text"
                  placeholder="Enter team number..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full rounded-md border-gray-700 bg-gray-800 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">
                  Filter by Drivetrain
                </label>
                <input
                  type="text"
                  placeholder="Enter drivetrain type..."
                  value={filterDrivetrain}
                  onChange={(e) => setFilterDrivetrain(e.target.value)}
                  className="w-full rounded-md border-gray-700 bg-gray-800 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">
                  Filter by Endgame
                </label>
                <select
                  value={filterEndgameType}
                  onChange={(e) => setFilterEndgameType(e.target.value)}
                  className="w-full rounded-md border-gray-700 bg-gray-800 text-white"
                >
                  <option value="">All</option>
                  <option value="deep">Deep</option>
                  <option value="shallow">Shallow</option>
                  <option value="none">None</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">
                  Filter by Auto Scoring
                </label>
                <select
                  value={filterAutoPosition}
                  onChange={(e) => setFilterAutoPosition(e.target.value)}
                  className="w-full rounded-md border-gray-700 bg-gray-800 text-white"
                >
                  <option value="">All</option>
                  <option value="coral">Auto Coral</option>
                  <option value="algae">Auto Algae</option>
                  <option value="both">Both</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-1">
                  Filter by Teleop Scoring Type
                </label>
                <select
                  value={filterTeleopPreference}
                  onChange={(e) => setFilterTeleopPreference(e.target.value)}
                  className="w-full rounded-md border-gray-700 bg-gray-800 text-white"
                >
                  <option value="">All</option>
                  <option value="coral">Coral Only</option>
                  <option value="algae">Algae Only</option>
                  <option value="both">Both</option>
                </select>
              </div>
            </div>
          </div>

          {filteredTeams?.length === 0 ? (
            <div className="p-8 text-center text-gray-400">
              No teams found matching your search criteria
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-700">
                <thead className="bg-gray-900">
                  <tr>
                    {columns.map((column) => (
                      <th
                        key={column.key}
                        className="px-6 py-3 text-left text-xs font-medium text-gray-200 uppercase tracking-wider"
                      >
                        {column.header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-gray-800 divide-y divide-gray-700">
                  {filteredTeams.map((team) => (
                    <tr
                      key={team.teamNumber}
                      onClick={() => navigate(`/team/${team.teamNumber}`)}
                      className="hover:bg-gray-700 cursor-pointer"
                    >
                      {columns.map((column) => (
                        <td
                          key={`${team.teamNumber}-${column.key}`}
                          className="px-6 py-4 whitespace-nowrap text-sm text-white"
                        >
                          {column.render
                            ? column.render(team[column.key as keyof ScoutingData], team)
                            : team[column.key as keyof ScoutingData]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 