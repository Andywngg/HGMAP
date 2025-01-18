import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import toast from 'react-hot-toast';
import endpoints from '../config/api';

interface TeamData {
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
  endgameType: string;
  robotWidth: number;
  robotLength: number;
  robotHeight: number;
  robotWeight: number;
  drivetrainType: string;
  imageUrl: string | null;
  mustStartSpecificPosition: boolean;
  teleopPreference: string;
  scoringPreference: string;
  coralLevels: string[];
}

const TeamDetails = () => {
  const { teamNumber } = useParams();
  const navigate = useNavigate();
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTeamData = async () => {
      try {
        if (!teamNumber) {
          toast.error('Invalid team number');
          navigate('/dashboard');
          return;
        }
        
        const response = await axios.get(endpoints.teams.get(teamNumber));
        
        if (response.data) {
          setTeamData(response.data);
        } else {
          throw new Error('Team not found');
        }
      } catch (error: any) {
        console.error('Error fetching team data:', error);
        toast.error(error.response?.data?.error || 'Failed to load team data');
        navigate('/dashboard');
      } finally {
        setLoading(false);
      }
    };

    fetchTeamData();
  }, [teamNumber, navigate]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-xl text-white">Loading...</div>
      </div>
    );
  }

  if (!teamData) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6 text-white">Team {teamData.teamNumber}</h1>
      
      <div className="bg-gray-800 rounded-lg shadow-xl p-6 space-y-6 text-white">
        {/* Auto */}
        <section>
          <h2 className="text-2xl font-semibold text-red-500 mb-4">Autonomous</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>Auto Score Coral: {teamData.autoScoreCoral ? '✓' : '✗'}</div>
            <div>Auto Score Algae: {teamData.autoScoreAlgae ? '✓' : '✗'}</div>
            <div>Must Start Specific Position: {teamData.mustStartSpecificPosition ? '✓' : '✗'}</div>
            {teamData.mustStartSpecificPosition && (
              <div>Starting Position: {teamData.autoStartingPosition}</div>
            )}
          </div>
        </section>

        {/* Teleop */}
        <section>
          <h2 className="text-2xl font-semibold text-red-500 mb-4">Teleop</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>Can Dealgify: {teamData.teleopDealgifying ? '✓' : '✗'}</div>
            <div>Teleop Preference: {teamData.teleopPreference}</div>
            <div>Scoring Preference: {teamData.scoringPreference}</div>
            <div>Coral Levels: {teamData.coralLevels?.join(', ') || 'None'}</div>
          </div>
        </section>

        {/* Endgame */}
        <section>
          <h2 className="text-2xl font-semibold text-red-500 mb-4">Endgame</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>Endgame Type: {teamData.endgameType}</div>
          </div>
        </section>

        {/* Robot Specs */}
        <section>
          <h2 className="text-2xl font-semibold text-red-500 mb-4">Robot Specifications</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>Width: {teamData.robotWidth} inches</div>
            <div>Length: {teamData.robotLength} inches</div>
            <div>Height: {teamData.robotHeight} inches</div>
            <div>Weight: {teamData.robotWeight} lbs</div>
            <div>Drivetrain: {teamData.drivetrainType}</div>
          </div>
        </section>

        {/* Robot Image */}
        {teamData.imageUrl && (
          <section>
            <h2 className="text-2xl font-semibold text-red-500 mb-4">Robot Image</h2>
            <img 
              src={endpoints.uploads.getUrl(teamData.imageUrl)} 
              alt={`Team ${teamData.teamNumber} Robot`}
              className="max-w-full h-auto rounded-lg shadow-lg"
            />
          </section>
        )}
      </div>

      <button
        onClick={() => navigate('/dashboard')}
        className="mt-6 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
      >
        Back to Dashboard
      </button>
    </div>
  );
};

export default TeamDetails; 