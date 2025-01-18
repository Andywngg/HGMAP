'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    await queryInterface.createTable('teams', {
      id: {
        type: Sequelize.INTEGER,
        autoIncrement: true,
        primaryKey: true,
      },
      teamNumber: {
        type: Sequelize.INTEGER,
        allowNull: false,
        unique: true,
      },
      autoScoreCoral: {
        type: Sequelize.BOOLEAN,
        defaultValue: false,
      },
      autoScoreAlgae: {
        type: Sequelize.BOOLEAN,
        defaultValue: false,
      },
      autoStartingPosition: {
        type: Sequelize.STRING,
      },
      teleopCoralCapability: {
        type: Sequelize.INTEGER,
        defaultValue: 0,
      },
      teleopAlgaeCapability: {
        type: Sequelize.INTEGER,
        defaultValue: 0,
      },
      teleopDealgifying: {
        type: Sequelize.BOOLEAN,
        defaultValue: false,
      },
      teleopCoralLevels: {
        type: Sequelize.ARRAY(Sequelize.STRING),
        defaultValue: [],
      },
      endgameDeepPerformance: {
        type: Sequelize.INTEGER,
        defaultValue: 0,
      },
      endgameShallowPerformance: {
        type: Sequelize.INTEGER,
        defaultValue: 0,
      },
      endgameClimbing: {
        type: Sequelize.BOOLEAN,
        defaultValue: false,
      },
      robotWidth: {
        type: Sequelize.FLOAT,
      },
      robotLength: {
        type: Sequelize.FLOAT,
      },
      robotHeight: {
        type: Sequelize.FLOAT,
      },
      robotWeight: {
        type: Sequelize.FLOAT,
      },
      drivetrainType: {
        type: Sequelize.STRING,
      },
      notes: {
        type: Sequelize.TEXT,
      },
      imageUrl: {
        type: Sequelize.STRING,
      },
      createdAt: {
        type: Sequelize.DATE,
        allowNull: false,
      },
      updatedAt: {
        type: Sequelize.DATE,
        allowNull: false,
      },
    });
  },

  async down(queryInterface, Sequelize) {
    await queryInterface.dropTable('teams');
  }
}; 