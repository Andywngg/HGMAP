'use strict';

const bcrypt = require('bcryptjs');

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    const password = await bcrypt.hash('otisit!!!', 10);

    // First, create the admin user
    await queryInterface.bulkInsert('users', [{
      name: 'Admin User',
      email: '1334admin@gmail.com',
      password,
      teamNumber: 1334,
      createdAt: new Date(),
      updatedAt: new Date()
    }]);

    // Then, create the demo team
    await queryInterface.bulkInsert('teams', [{
      teamNumber: 1334,
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
      updatedAt: new Date()
    }]);
  },

  async down(queryInterface, Sequelize) {
    // Remove seeded data in reverse order
    await queryInterface.bulkDelete('teams', null, {});
    await queryInterface.bulkDelete('users', null, {});
  }
}; 