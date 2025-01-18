'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    await queryInterface.dropTable('teams', { cascade: true });
    
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
        allowNull: true,
      },
      teleopDealgifying: {
        type: Sequelize.BOOLEAN,
        defaultValue: false,
      },
      teleopPreference: {
        type: Sequelize.STRING,
        allowNull: true,
      },
      scoringPreference: {
        type: Sequelize.STRING,
        allowNull: true,
      },
      coralLevels: {
        type: Sequelize.ARRAY(Sequelize.STRING),
        defaultValue: [],
      },
      endgameType: {
        type: Sequelize.STRING,
        defaultValue: 'none',
      },
      robotWidth: {
        type: Sequelize.FLOAT,
        allowNull: true,
      },
      robotLength: {
        type: Sequelize.FLOAT,
        allowNull: true,
      },
      robotHeight: {
        type: Sequelize.FLOAT,
        allowNull: true,
      },
      robotWeight: {
        type: Sequelize.FLOAT,
        allowNull: true,
      },
      drivetrainType: {
        type: Sequelize.STRING,
        allowNull: true,
      },
      notes: {
        type: Sequelize.TEXT,
        allowNull: true,
      },
      imageUrl: {
        type: Sequelize.STRING,
        allowNull: true,
      },
      mustStartSpecificPosition: {
        type: Sequelize.BOOLEAN,
        defaultValue: false,
      },
      createdAt: {
        type: Sequelize.DATE,
        allowNull: false,
      },
      updatedAt: {
        type: Sequelize.DATE,
        allowNull: false,
      }
    });
  },

  async down(queryInterface, Sequelize) {
    await queryInterface.dropTable('teams');
  }
}; 