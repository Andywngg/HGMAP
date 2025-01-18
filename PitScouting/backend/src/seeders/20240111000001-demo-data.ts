import { QueryInterface } from 'sequelize';
import bcrypt from 'bcryptjs';

export async function up(queryInterface: QueryInterface) {
  const password = await bcrypt.hash('password123', 10);

  await queryInterface.bulkInsert('users', [
    {
      name: 'Demo User',
      email: 'demo@example.com',
      password,
      teamNumber: 1234,
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ]);

  await queryInterface.bulkInsert('teams', [
    {
      teamNumber: 1234,
      autoScoreCoral: true,
      autoScoreAlgae: false,
      autoStartingPosition: 'Left',
      teleopCoralCapability: 4,
      teleopAlgaeCapability: 3,
      teleopDealgifying: true,
      teleopCoralLevels: ['low', 'mid'],
      endgameDeepPerformance: 5,
      endgameShallowPerformance: 4,
      endgameClimbing: true,
      robotWidth: 30,
      robotLength: 32,
      robotHeight: 24,
      robotWeight: 120,
      drivetrainType: 'Swerve',
      notes: 'Demo team data',
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ]);
}

export async function down(queryInterface: QueryInterface) {
  await queryInterface.bulkDelete('users', {});
  await queryInterface.bulkDelete('teams', {});
} 