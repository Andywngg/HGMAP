'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    const tableInfo = await queryInterface.describeTable('teams');
    
    if (!tableInfo.teleopPreference) {
      await queryInterface.addColumn('teams', 'teleopPreference', {
        type: Sequelize.STRING,
        allowNull: true
      });
    }

    if (!tableInfo.scoringPreference) {
      await queryInterface.addColumn('teams', 'scoringPreference', {
        type: Sequelize.STRING,
        allowNull: true
      });
    }

    if (!tableInfo.coralLevels) {
      await queryInterface.addColumn('teams', 'coralLevels', {
        type: Sequelize.ARRAY(Sequelize.STRING),
        defaultValue: [],
        allowNull: true
      });
    }
  },

  async down(queryInterface, Sequelize) {
    try {
      await queryInterface.removeColumn('teams', 'teleopPreference');
      await queryInterface.removeColumn('teams', 'scoringPreference');
      await queryInterface.removeColumn('teams', 'coralLevels');
    } catch (error) {
      console.log('Error removing columns:', error);
    }
  }
}; 