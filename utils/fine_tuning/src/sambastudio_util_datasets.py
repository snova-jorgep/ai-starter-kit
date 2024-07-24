import os
import json
import yaml
from snsdk import SnSdk
from typing import Optional, List
import subprocess
import logging
import re

SNAPI_PATH = "~/.snapi"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

from dotenv import load_dotenv

load_dotenv("../.env", override=True)

JOB_TYPES = [
    "train",
    "evaluation",
    "batch_predict",
]  # can not combine train/evaluation with batch_predict
SOURCE_TYPES = ["local", "aws", "localMachine"]
SOURCE_FILE_PATH = "./fine_tuning/src/source_file.json"


class SnsdkWrapper:
    """init"""

    def __init__(self, config_path=None) -> None:

        self.config = None
        self.tenant_name = None
        self.snapi_path = SNAPI_PATH

        # If config is provided, load it and validate Snapi directory path
        if config_path is not None:
            self.config = self._load_config(config_path)
            config_snapi_path = self.config["sambastudio"]["snapi_path"]
            if config_snapi_path is not None and len(config_snapi_path) > 0:
                self.snapi_path = self.config["sambastudio"]["snapi_path"]

        # Get sambastudio variables to set up Snsdk
        host_url, tenant_id, access_key = self._get_sambastudio_variables()
        self.snsdk_client = SnSdk(
            host_url=host_url,
            access_key=access_key,
            tenant_id=tenant_id,
        )

    """Tenant methods"""

    def list_tenants(self, verbose: bool = False) -> Optional[str]:
        """Lists all tenants

        Returns:
            list | None: list of existing tenants. If there's an error, None is returned.
        """
        list_tenant_response = self.snsdk_client.list_tenants()
        if list_tenant_response["status_code"] == 200:
            tenants = []
            if verbose:
                return list_tenant_response["data"]
            else:
                for tenant in list_tenant_response["data"]:
                    tenants.append(
                        {
                            "tenant_id": tenant.get("tenant_id"),
                            "tenant_name": tenant.get("tenant_name"),
                        }
                    )
                return tenants
        else:
            logging.error(
                f"Failed to list projects. Details: {list_tenant_response['detail']}"
            )
            raise Exception(f"Error message: {list_tenant_response['detail']}")

    def search_tenant(self, tenant_name: Optional[str]) -> Optional[str]:
        """Searches tenant

        Args:
            tenant_name (str): tenant name to search

        Returns:
            str | None: searched tenant information. If there's an error, None is returned.
        """

        tenant_info_response = self.snsdk_client.tenant_info(tenant=tenant_name)
        if tenant_info_response["status_code"] == 200:
            tenant_id = tenant_info_response["data"]["tenant_id"]
            logging.info(f"Tenant with name '{tenant_name}' found with id {tenant_id}")
            return tenant_id
        else:
            logging.info(f"Tenant with name '{tenant_name}' not found")
            return None

    """Internal methods"""

    def _set_snsdk_using_env_variables(
        self,
        host_name: str,
        access_key: str,
        tenant_name: str,
        current_snapi_config: dict,
        snapi_config_path: str,
        snapi_secret_path: str,
    ) -> tuple:
        """Sets Snsdk using env variables. It also validates if tenant can be set in Snapi config file.
        Args:
            host_name (str): host name coming from env variables
            access_key (str): access key coming from env variables
            tenant_name (str): tenant name coming from env variables
            current_snapi_config (dict): current snapi config dictionary
            snapi_config_path (str): snapi config path
            snapi_secret_path (str): snapi secret path

        Raises:
            Exception: fails to set the specified tenant using Snapi CLI

        Returns:
            tuple: host name, access key and tenant id
        """
        # Updates snapi config file using requested Sambastudio env
        tmp_snapi_config = {}
        tmp_snapi_config["HOST_NAME"] = host_name
        tmp_snapi_config["CONFIG_DIR"] = current_snapi_config["CONFIG_DIR"]
        tmp_snapi_config["DISABLE_SSL_WARNINGS"] = current_snapi_config[
            "DISABLE_SSL_WARNINGS"
        ]
        with open(snapi_config_path, "w") as file:
            json.dump(tmp_snapi_config, file)
        with open(snapi_secret_path, "w") as file:
            file.write(access_key)

        # Sets default requested tenant
        snapi_config_response = subprocess.run(
            ["snapi", "config", "set", "--tenant", f"{tenant_name}"],
            capture_output=True,
            text=True,
        )

        # If there's an error in Snapi subprocess, show it and stop process
        if (
            ("status_code" in snapi_config_response.stdout.lower())
            and ("error occured" in snapi_config_response.stdout.lower())
            or (len(snapi_config_response.stderr) > 0)
        ):
            if len(snapi_config_response.stderr) > 0:
                error_message = snapi_config_response.stderr
            else:
                error_message = re.search(
                    r"message:\s*(.*)", snapi_config_response.stdout
                )[0]
            logging.error(
                f"Failed to set tenant with name '{tenant_name}'. Details: {error_message}"
            )
            raise Exception(f"Error message: {error_message}")

        # Read updated Snapi config file
        with open(snapi_config_path, "r") as file:
            new_snapi_config = json.load(file)

        return host_name, new_snapi_config["TENANT_ID"], access_key

    def _get_sambastudio_variables(self) -> tuple:
        """Gets Sambastudio host name, tenant id and access key from Snapi folder location

        Raises:
            FileNotFoundError: raises error when the snapi config or secret file is not found
            ValueError: raises error when the snapi config file doesn't contain a correct json format

        Returns:
            tuple: host name, tenant id and access key from snapi setup
        """
        snapi_config = ""
        snapi_secret = ""

        try:

            # reads snapi config json
            snapi_config_path = os.path.expanduser(self.snapi_path) + "/config.json"
            with open(snapi_config_path, "r") as file:
                snapi_config = json.load(file)

            # reads snapi secret txt file
            snapi_secret_path = os.path.expanduser(self.snapi_path) + "/secret.txt"
            with open(snapi_secret_path, "r") as file:
                snapi_secret = file.read()

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The file {snapi_config_path} does not exist."
            )
        except ValueError:
            raise ValueError(
                f"Error: The file {snapi_config_path} contains invalid JSON."
            )

        host_name = os.getenv("SAMBASTUDIO_HOST_NAME")
        access_key = os.getenv("SAMBASTUDIO_ACCESS_KEY")
        tenant_name = os.getenv("SAMBASTUDIO_TENANT_NAME")

        if (
            (host_name is not None)
            and (access_key is not None)
            and (tenant_name is not None)
        ):
            logging.info(f"Using env variables to set up Snsdk.")
            host_name, tenant_id, access_key = self._set_snsdk_using_env_variables(
                host_name,
                access_key,
                tenant_name,
                snapi_config,
                snapi_config_path,
                snapi_secret_path,
            )

        else:
            logging.info(f"Using variables from Snapi config to set up Snsdk.")
            host_name = snapi_config["HOST_NAME"]
            tenant_id = snapi_config["TENANT_ID"]
            access_key = snapi_secret

        return host_name, tenant_id, access_key

    def _load_config(self, file_path):
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
            logging.info(f"Using config file located in {file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
        except yaml.scanner.ScannerError:
            raise ValueError(f"Error: The file {file_path} contains invalid yaml.")
        return config

    def _raise_error_if_config_is_none(self) -> None:
        if self.config is None:
            error_message = "No config found. Please provide parameter values."
            logging.error(error_message)
            raise Exception(f"Error message: {error_message}")

    """Dataset"""

    def list_datasets(self, verbose: bool = False) -> Optional[list]:
        """Lists all datasets

        Returns:
            list | None: list of existing datasets. If there's an error, None is returned.
        """
        list_datasets_response = self.snsdk_client.list_datasets()
        if list_datasets_response["status_code"] == 200:
            datasets = []
            if verbose:
                return list_datasets_response["datasets"]
            else:
                for dataset in list_datasets_response["datasets"]:
                    datasets.append(
                        {
                            "id": dataset.get("id"),
                            "dataset_name": dataset.get("dataset_name"),
                        }
                    )
                return datasets
        else:
            logging.error(
                f"Failed to list models. Details: {list_datasets_response['detail']}"
            )
            raise Exception(f"Error message: {list_datasets_response['detail']}")

    def search_dataset(self, dataset_name: Optional[str] = None) -> Optional[str]:
        """Searches a dataset

        Args:
            dataset_name (str): dataset name to search

        Returns:
            str | None: searched dataset information. If there's an error, None is returned.
        """
        # Decide whether using method parameters or config
        if dataset_name is None:
            self._raise_error_if_config_is_none()
            dataset_name = self.config["dataset"]["dataset_name"]

        # Search dataset
        search_dataset_response = self.snsdk_client.search_dataset(
            dataset_name=dataset_name
        )
        if search_dataset_response["status_code"] == 200:
            dataset_id = search_dataset_response["data"]["dataset_id"]
            logging.info(
                f"Dataset with name '{dataset_name}' found with id {dataset_id}"
            )
            return dataset_id
        else:
            logging.info(f"Dataset with name '{dataset_name}' not found")
            return None

    def delete_dataset(self, dataset_name: Optional[str] = None) -> None:
        # Decide whether using method parameters or config
        if dataset_name is None:
            self._raise_error_if_config_is_none()
            dataset_name = self.config["dataset"]["dataset_name"]

        # Search dataset if exists
        dataset_id = self.search_dataset(dataset_name=dataset_name)
        if dataset_id is None:
            raise Exception(f"Dataset with name '{dataset_name}' not found")

        # Delete dataset
        delete_dataset_response = self.snsdk_client.delete_dataset(dataset=dataset_name)
        if delete_dataset_response["status_code"] == 200:
            logging.info(f"Dataset with name '{dataset_name}' deleted")
        else:
            logging.error(
                f"Failed to delete dataset with name '{dataset_name}'. Details: {delete_dataset_response}"
            )
            raise Exception(f"Error message: {delete_dataset_response}")

    def _create_source_file(self, dataset_path) -> None:
        json_content = {"source_path": dataset_path}
        with open(SOURCE_FILE_PATH, "w") as file:
            json.dump(json_content, file)

    def _build_snapi_dataset_add_command(
        self,
        dataset_name,
        dataset_apps_availability,
        dataset_job_types,
        dataset_source_type,
        dataset_description,
        dataset_filetype,
        dataset_url,
        dataset_language,
    ):
        # Get multiple job type parameters
        job_type_command_parameters = []
        for job_type in dataset_job_types:
            job_type_command_parameters.append("--job_type")
            job_type_command_parameters.append(job_type)

        # Get multiple apps parameters
        apps_command_parameters = []
        for app in dataset_apps_availability:
            apps_command_parameters.append("--apps")
            apps_command_parameters.append(app)

        command = [
            "snapi",
            "dataset",
            "add",
            "--dataset-name",
            dataset_name,
            "--description",
            dataset_description,
            "--source_type",
            dataset_source_type,
            "--language",
            dataset_language,
            "--source_file",
            SOURCE_FILE_PATH,
            "--file_type",
            dataset_filetype,
            "--url",
            dataset_url,
        ]
        command.extend(job_type_command_parameters)
        command.extend(apps_command_parameters)

        return command

    def create_dataset(
        self,
        # snapi required parameters
        dataset_name: Optional[str] = None,
        dataset_apps_availability: Optional[List[str]] = None,
        dataset_job_types: Optional[List[str]] = None,
        dataset_source_type: Optional[str] = None,
        dataset_path: Optional[str] = None,
        # non-required paramaters
        # TODO: add metadata file paths
        # dataset_metadata_file: Optional[str] = None,
        dataset_description: Optional[str] = None,
        dataset_filetype: Optional[str] = None,
        dataset_url: Optional[str] = None,
        dataset_language: Optional[str] = None,
    ):
        # Decide whether using method parameters or config
        if dataset_name is None:
            self._raise_error_if_config_is_none()
            dataset_name = self.config["dataset"]["dataset_name"]

        # Validate if apps exist
        if dataset_apps_availability is None:
            self._raise_error_if_config_is_none()
            dataset_apps_availability = self.config["dataset"][
                "dataset_apps_availability"
            ]
        for app_name in dataset_apps_availability:
            app_id = self.search_app(app_name)
            if app_id is None:
                raise Exception(f"App '{app_name}' not found")

        # Validate job types
        if dataset_job_types is None:
            self._raise_error_if_config_is_none()
            dataset_job_types = self.config["dataset"]["dataset_job_types"]
        for job_type in dataset_job_types:
            if job_type not in JOB_TYPES:
                raise Exception(f"Job type '{job_type}' not valid")

        # Validate source type
        if dataset_source_type is None:
            self._raise_error_if_config_is_none()
            dataset_source_type = self.config["dataset"]["dataset_source_type"]
        if dataset_source_type not in SOURCE_TYPES:
            raise Exception(f"Source type '{dataset_source_type}' not valid")

        # Decide whether using method parameters or config
        if dataset_path is None:
            self._raise_error_if_config_is_none()
            dataset_path = self.config["dataset"]["dataset_path"]

        # Create source file based on dataset path
        self._create_source_file(dataset_path)

        # Decide whether using method parameters or config
        if dataset_description is None:
            self._raise_error_if_config_is_none()
            dataset_description = self.config["dataset"]["dataset_description"]

        if dataset_filetype is None:
            self._raise_error_if_config_is_none()
            dataset_filetype = self.config["dataset"]["dataset_filetype"]

        if dataset_url is None:
            self._raise_error_if_config_is_none()
            dataset_url = self.config["dataset"]["dataset_url"]

        if dataset_language is None:
            self._raise_error_if_config_is_none()
            dataset_language = self.config["dataset"]["dataset_language"]

        # Validate if dataset already exists
        dataset_id = self.search_dataset(dataset_name)

        # Create dataset if dataset is not found
        if dataset_id is None:
            command = self._build_snapi_dataset_add_command(
                dataset_name,
                dataset_apps_availability,
                dataset_job_types,
                dataset_source_type,
                dataset_description,
                dataset_filetype,
                dataset_url,
                dataset_language,
            )
            echo_response = subprocess.run(
                ["echo", "yes"], capture_output=True, text=True
            )
            snapi_response = subprocess.run(
                command, input=echo_response.stdout, capture_output=True, text=True
            )

            # Check if possible errors in response
            errors_found_in_response = (
                ("status_code" in snapi_response.stdout.lower())
                and ("error occured" in snapi_response.stdout.lower())
            ) or (len(snapi_response.stderr) > 0)
            if errors_found_in_response:
                if len(snapi_response.stderr) > 0:
                    error_message = snapi_response.stderr
                else:
                    error_message = re.search(
                        r"message:\s*(.*)", snapi_response.stdout
                    )[0]
                logging.error(
                    f"Failed to create dataset with name '{dataset_name}'. Details: {error_message}"
                )
                raise Exception(f"Error message: {error_message}")
            else:
                dataset_id = self.search_dataset(dataset_name=dataset_name)
                logging.info(
                    f"Dataset with name '{dataset_name}' created: '{snapi_response.stdout}'"
                )
                return dataset_id
        else:
            logging.info(
                f"Dataset with name '{dataset_name}' already exists with id '{dataset_id}', using it"
            )
        return dataset_id

    """app"""

    def list_apps(self, verbose: bool = False) -> list | None:
        """Lists all apps

        Returns:
            list | None: list of existing apps. If there's an error, None is returned.
        """
        list_apps_response = self.snsdk_client.list_apps()
        if list_apps_response["status_code"] == 200:
            apps = []
            if verbose:
                apps = list_apps_response["apps"]
            else:
                for app in list_apps_response["apps"]:
                    apps.append({"id": app.get("id"), "name": app.get("name")})
            return apps
        else:
            logging.error(
                f"Failed to list models. Details: {list_apps_response['detail']}"
            )
            raise Exception(f"Error message: {list_apps_response['detail']}")

    def search_app(self, app_name: str) -> str | None:
        """Searches an App

        Args:
            app_name (str): app name to search

        Returns:
            str | None: searched app information. If there's an error, None is returned.
        """
        app_info_response = self.snsdk_client.app_info(app=app_name)
        if app_info_response["status_code"] == 200:
            app_id = app_info_response["apps"]["id"]
            logging.info(f"App with name '{app_name}' found with id {app_id}")
            return app_id
        else:
            logging.info(f"App with name '{app_name}' not found")
            return None


if __name__ == "__main__":

    # config_path = None
    config_path = "./fine_tuning/config.yaml"
    snsdkwrapper_client = SnsdkWrapper(config_path=config_path)

    # list tenants
    # tenants = snsdkwrapper_client.list_tenants()
    # tenants = snsdkwrapper_client.list_tenants(verbose=True)
    # print(tenants)

    # search tenant
    # existing tenant
    # tenant = snsdkwrapper_client.search_tenant(tenant_name="cap-engagements")
    # non-existing tenant
    # tenant = snsdkwrapper_client.search_tenant(tenant_name="my_dummy_tenant")
    # print(tenant)

    # list datasets
    # datasets = snsdkwrapper_client.list_datasets()
    # datasets = snsdkwrapper_client.list_datasets(verbose=True)
    # print(datasets[:2])

    # search dataset
    # existing dataset
    # dataset = snsdkwrapper_client.search_dataset(dataset_name="MITRE_test_rm")
    # non-existing dataset
    # dataset = snsdkwrapper_client.search_dataset(dataset_name="my_dummy_dataset")
    # using config
    # dataset = snsdkwrapper_client.search_dataset()
    # print(dataset)

    # create dataset
    # new dataset
    # dataset = snsdkwrapper_client.create_dataset(
    #     dataset_name="mitre dataset test_9",
    #     dataset_apps_availability=["E5 Mistral Embedding", "Text Embedding"],
    #     dataset_job_types=["batch_predict"],
    #     dataset_source_type="localMachine",
    #     dataset_path="/Users/rodrigom/Desktop/aisk/finetuning/mitre",
    #     # dataset_metadata_file: Optional[str] = None,
    #     dataset_description="test",
    #     dataset_filetype="hdf5",
    #     dataset_url="",
    #     dataset_language="english",
    # )
    # dataset already exists
    # dataset = snsdkwrapper_client.create_dataset(
    #     dataset_name="mitre dataset test_4",
    #     dataset_apps_availability=["E5 Mistral Embedding", "Text Embedding"],
    #     dataset_job_types=["batch_predict"],
    #     dataset_source_type="localMachine",
    #     dataset_path="/Users/rodrigom/Desktop/aisk/finetuning/mitre",
    #     # dataset_metadata_file: Optional[str] = None,
    #     dataset_description="test",
    #     dataset_filetype="hdf5",
    #     dataset_url="",
    #     dataset_language="english",
    # )
    # using config
    # dataset = snsdkwrapper_client.create_dataset()
    # print(dataset)

    # delete dataset
    # existing dataset
    # snsdkwrapper_client.delete_dataset(dataset_name="MITRE_test_rm") # not authorized for user
    # non-existing dataset
    # snsdkwrapper_client.delete_dataset(dataset_name="my_dummy_dataset")
    # using config
    # snsdkwrapper_client.delete_dataset()  # not authorized for user

    # list apps
    # apps = snsdkwrapper_client.list_apps()
    # apps = snsdkwrapper_client.list_apps(verbose=True)
    # print(apps[:2])

    # search app
    # existing app
    # app = snsdkwrapper_client.search_app(app_name="E5 Mistral Embedding")
    # non-existing app
    # app = snsdkwrapper_client.search_app(app_name="my_dummy_app")
    # print(app)

    # print(snsdkwrapper_client.list_apps()[0])
    # print(app_name)
    # print(app_id)
