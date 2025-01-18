import { Request, Response } from 'express';
import { Team } from '../models';
import { Op } from 'sequelize';

export const createTeam = async (req: Request, res: Response): Promise<void> => {
  try {
    console.log('Received team data:', JSON.stringify(req.body, null, 2));
    
    // Validate teamNumber
    if (!req.body.teamNumber) {
      throw new Error('Team number is required');
    }

    // Parse and validate numeric values first
    const teamNumber = parseInt(req.body.teamNumber);
    const robotWidth = req.body.robotWidth ? parseFloat(req.body.robotWidth) : null;
    const robotLength = req.body.robotLength ? parseFloat(req.body.robotLength) : null;
    const robotHeight = req.body.robotHeight ? parseFloat(req.body.robotHeight) : null;
    const robotWeight = req.body.robotWeight ? parseFloat(req.body.robotWeight) : null;

    if (isNaN(teamNumber)) {
      throw new Error('Invalid team number format');
    }

    // Validate numeric values
    [robotWidth, robotLength, robotHeight, robotWeight].forEach((value, index) => {
      if (value !== null && isNaN(value)) {
        throw new Error(`Invalid numeric value for ${['width', 'length', 'height', 'weight'][index]}`);
      }
    });

    let coralLevels = [];
    try {
      if (typeof req.body.coralLevels === 'string') {
        coralLevels = JSON.parse(req.body.coralLevels);
      } else if (Array.isArray(req.body.coralLevels)) {
        coralLevels = req.body.coralLevels;
      }
    } catch (error) {
      console.error('Error parsing coralLevels:', error);
      coralLevels = [];
    }
    
    const processedData = {
      teamNumber,
      autoScoreCoral: req.body.autoScoreCoral === 'true',
      autoScoreAlgae: req.body.autoScoreAlgae === 'true',
      mustStartSpecificPosition: req.body.mustStartSpecificPosition === 'true',
      autoStartingPosition: req.body.autoStartingPosition || null,
      teleopDealgifying: req.body.teleopDealgifying === 'true',
      teleopPreference: req.body.teleopPreference || null,
      scoringPreference: req.body.scoringPreference || null,
      coralLevels,
      endgameType: req.body.endgameType || 'none',
      robotWidth,
      robotLength,
      robotHeight,
      robotWeight,
      drivetrainType: req.body.drivetrainType || null,
      notes: req.body.notes || '',
      imageUrl: req.file ? `/uploads/${req.file.filename}` : null,
    };

    console.log('Processed team data:', JSON.stringify(processedData, null, 2));

    try {
      const team = await Team.create(processedData);
      console.log('Team created successfully:', team.toJSON());
      res.status(201).json(team);
    } catch (dbError: any) {
      console.error('Database error:', dbError);
      console.error('SQL Query:', dbError.sql);
      console.error('SQL Parameters:', dbError.parameters);
      throw dbError;
    }
  } catch (error: any) {
    console.error('Error creating team:', error);
    console.error('Error details:', error.original || error);
    console.error('Error stack:', error.stack);
    res.status(400).json({ 
      error: 'Error creating team', 
      message: error.message,
      details: error.original?.detail || error.original?.message || error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
};

export const getTeam = async (req: Request, res: Response): Promise<Response> => {
  try {
    const teamNumber = parseInt(req.params.teamNumber);
    
    if (isNaN(teamNumber)) {
      return res.status(400).json({ error: 'Invalid team number' });
    }

    const team = await Team.findOne({
      where: { teamNumber },
    });

    if (!team) {
      return res.status(404).json({ error: 'Team not found' });
    }

    return res.json(team);
  } catch (error: any) {
    console.error('Error fetching team:', error);
    return res.status(500).json({ error: 'Error fetching team', details: error.message });
  }
};

export const updateTeam = async (req: Request, res: Response): Promise<Response> => {
  try {
    const team = await Team.findOne({
      where: { teamNumber: req.params.teamNumber },
    });

    if (!team) {
      return res.status(404).json({ error: 'Team not found' });
    }

    await team.update(req.body);
    return res.json(team);
  } catch (error: any) {
    console.error('Error updating team:', error);
    return res.status(400).json({ error: 'Error updating team', details: error.message });
  }
};

export const getAllTeams = async (req: Request, res: Response): Promise<void> => {
  try {
    const { search, drivetrain, endgameType, autoPosition, teleopPreference } = req.query;
    const where: any = {};

    if (search) {
      where.teamNumber = { [Op.like]: `%${search}%` };
    }

    if (drivetrain) {
      where.drivetrainType = { [Op.iLike]: `%${drivetrain}%` };
    }

    if (endgameType) {
      where.endgameType = endgameType;
    }

    if (autoPosition) {
      where.autoStartingPosition = autoPosition;
    }

    if (teleopPreference) {
      where.teleopPreference = teleopPreference;
    }

    const teams = await Team.findAll({ where });
    res.json(teams);
  } catch (error: any) {
    console.error('Error fetching teams:', error);
    res.status(400).json({ error: 'Error fetching teams', details: error.message });
  }
}; 